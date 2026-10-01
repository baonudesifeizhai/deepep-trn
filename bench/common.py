"""Shared benchmark plumbing: presets, routing, timing, byte accounting, run metadata.

Timing convention (all benches): synchronized host wall-clock. Each round is
barrier -> N calls -> device sync; the round time is the max across ranks
(critical path) divided by N; we report median/min/max over rounds.

Two flavours for device-resident calls:
  pipelined (default): calls are issued back to back, one sync per round, so
      launch overhead overlaps execution -- an amortized throughput number;
  latency (sync_each=True): sync after every call -- what a decode step that
      waits on each layer's communication sees.
"""
from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from deep_ep_trn.testing import make_routing, strided_ep_groups  # noqa: F401  (re-exported for benches)

# (hidden, num_experts, top_k)
PRESETS = {
    'qwen3-30b-a3b': (2048, 128, 8),
    'qwen3-235b-a22b': (4096, 128, 8),
    'deepseek-v3': (7168, 256, 8),
}
ROUTINGS = ('random', 'balanced', 'zipf', 'hot_rank0')  # kinds that keep T fixed


def csv_list(kind):
    return lambda s: [kind(v) for v in s.split(',') if v]


def base_parser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument('--backend', choices=['nki', 'gloo'], default='nki')
    p.add_argument('--ep', type=int, default=4)
    p.add_argument('--presets', type=csv_list(str), default=['qwen3-30b-a3b'])
    p.add_argument('--tokens', type=csv_list(int), default=[16, 32, 64, 128], help='tokens per rank')
    p.add_argument('--rounds', type=int, default=5)
    p.add_argument('--max-iters', type=int, default=50)
    p.add_argument('--target-round-s', type=float, default=0.5,
                   help='adaptive iterations: aim for this much time per round')
    p.add_argument('--out-dir', type=Path, default=ROOT / 'results' / 'adhoc')
    return p


class Ctx:
    """Process-wide distributed state for one EP layout."""

    def __init__(self, backend: str, ep: int):
        self.backend = backend
        if backend == 'nki':
            os.environ.setdefault('NEURON_PLATFORM_TARGET_OVERRIDE', 'trn2')
            import torch_neuronx  # noqa: F401
            dist.init_process_group('neuron')
            self.device = torch.device('neuron', int(os.environ.get('LOCAL_RANK', 0)))
            self.sync = torch.neuron.synchronize
        else:
            dist.init_process_group('gloo')
            self.device = torch.device('cpu')
            self.sync = lambda: None
        torch.set_num_threads(1)
        self.rank, self.world = dist.get_rank(), dist.get_world_size()
        self.control = dist.new_group(backend='gloo')
        self.groups = strided_ep_groups(self.world, ep)
        self.ep = ep
        for ranks in self.groups:  # every rank must create every group, in order
            g = dist.new_group(ranks=ranks, backend='gloo')
            n = dist.new_group(ranks=ranks, backend='neuron') if backend == 'nki' else None
            if self.rank in ranks:
                self.ep_group, self.ep_device_group, self.peers = g, n, ranks
        self.ep_rank = self.peers.index(self.rank)
        from deep_ep_trn.transport import GlooTransport, NkiA2avTransport
        self.transport = (NkiA2avTransport(self.groups, self.rank) if backend == 'nki'
                          else GlooTransport(self.ep_group))

    def buffer(self, num_max_tokens_per_rank: int):
        from deep_ep_trn import Buffer
        return Buffer(self.ep_group, transport=self.transport, num_max_tokens_per_rank=num_max_tokens_per_rank)

    def to_dev(self, t: torch.Tensor) -> torch.Tensor:
        return t.to(self.device)

    def gather(self, obj):
        out = [None] * self.world
        dist.all_gather_object(out, obj, group=self.control)
        return out

    def peer_values(self, obj):
        """Values from the ranks of this EP group, in group order."""
        data = dict(self.gather((self.rank, obj)))
        return [data[p] for p in self.peers]

    def measure(self, call, rounds: int, max_iters: int, target_round_s: float,
                sync_each: bool = False) -> dict:
        for _ in range(2):
            call()
        self.sync()
        start = time.perf_counter()
        call()
        self.sync()
        one = max(self.gather(time.perf_counter() - start))
        iters = int(max(2, min(max_iters, target_round_s / max(one, 1e-6))))
        per_round = []
        for _ in range(rounds):
            dist.barrier(group=self.control)
            start = time.perf_counter()
            for _ in range(iters):
                call()
                if sync_each:
                    self.sync()
            self.sync()
            per_round.append(max(self.gather((time.perf_counter() - start) * 1e6 / iters)))
        return dict(median_us=round(statistics.median(per_round), 1), min_us=round(min(per_round), 1),
                    max_us=round(max(per_round), 1), iters=iters, rounds=rounds)

    def first_call(self, call):
        """Run once (cold: includes NEFF compile on an empty cache); returns (output, seconds)."""
        start = time.perf_counter()
        out = call()
        self.sync()
        return out, max(self.gather(time.perf_counter() - start))

    def finish(self):
        dist.barrier(group=self.control)
        dist.destroy_process_group()


def gbps(num_bytes: float, us: float) -> float:
    return round(num_bytes / (us * 1e-6) / 1e9, 3) if us > 0 else 0.0


def capture_env(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / 'env.json'
    if path.exists():
        return

    def run(*cmd):
        try:
            return subprocess.run(cmd, text=True, capture_output=True, timeout=30).stdout.strip()
        except Exception as exc:  # tools may be missing in some containers
            return f'unavailable: {exc}'

    versions = {}
    for pkg in ('torch', 'torch-neuronx', 'neuronx-cc', 'nki'):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            versions[pkg] = None
    neuron_ls = run('neuron-ls')
    hw = {k: line.split(':', 1)[1].strip() for line in neuron_ls.splitlines()
          for k in ('instance-type', 'logical-neuroncore-config') if line.startswith(k)}
    env = dict(
        captured_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds'),
        python=sys.version.split()[0], versions=versions, hardware=hw,
        runtime_packages=run('dpkg-query', '-W', 'aws-neuronx-runtime-lib', 'aws-neuronx-collectives'),
        deep_ep_trn_git=dict(head=run('git', '-C', str(ROOT), 'rev-parse', '--short', 'HEAD'),
                             dirty=bool(run('git', '-C', str(ROOT), 'status', '--porcelain'))),
        env={k: os.environ.get(k) for k in ('NEURON_RT_VISIBLE_CORES', 'NEURON_PLATFORM_TARGET_OVERRIDE',
                                            'NEURON_LOGICAL_NC_CONFIG', 'OMP_NUM_THREADS')},
    )
    path.write_text(json.dumps(env, indent=2) + '\n')


def append_jsonl(out_dir: Path, name: str, record: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / name, 'a') as f:
        f.write(json.dumps(record) + '\n')
