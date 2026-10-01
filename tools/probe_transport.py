"""Probe which fixed-slot shapes the NKI a2av transport accepts on this runtime.

torchrun --nproc-per-node=4 tools/probe_transport.py --ep 4 --short-slots 1
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
parser = argparse.ArgumentParser()
parser.add_argument('--ep', type=int, choices=[2, 4], default=4)
parser.add_argument('--short-slots', type=int, choices=[0, 1], default=1)
parser.add_argument('--rows', type=int, default=48, help='max rows sent to one peer')
parser.add_argument('--width', type=int, default=2112)
args = parser.parse_args()

import torch
import torch.distributed as dist
from deep_ep_trn.transport import NkiA2avTransport

rank = int(os.environ['RANK'])
dist.init_process_group('neuron')
control = dist.new_group(backend='gloo')
groups = [[0, 1, 2, 3]] if args.ep == 4 else [[0, 2], [1, 3]]
peers = next(g for g in groups if rank in g)
t = NkiA2avTransport(groups, rank, short_slots=bool(args.short_slots))
cap = args.rows * t.num_ranks
slot = t.slot_rows(cap, args.rows)
g = torch.Generator().manual_seed(rank)
counts = torch.randint(0, args.rows + 1, (t.num_ranks,), generator=g)
counts[(t.rank + 1) % t.num_ranks] = 0      # exercise an empty peer
counts[t.rank] = args.rows                  # and a full one
send = torch.randn(cap, args.width, generator=g).to(torch.bfloat16)
everyone = [None] * dist.get_world_size()
dist.all_gather_object(everyone, (rank, send, counts), group=control)
src_data = {r: (s, c) for r, s, c in everyone if r in peers}
status = dict(rank=rank, ep=args.ep, short_slots=bool(args.short_slots), cap=cap, slot=slot)
try:
    recv, recv_counts = t.all_to_all_fixed_slot(send, counts, slot)
    for s, p in enumerate(peers):
        psend, pcounts = src_data[p]
        n, off = int(pcounts[t.rank]), int(pcounts[:t.rank].sum())
        assert int(recv_counts[s]) == n, ('count', p, int(recv_counts[s]), n)
        assert torch.equal(recv[s * slot:s * slot + n].view(torch.int16),
                           psend[off:off + n].view(torch.int16)), ('payload', p)
    status['result'] = 'pass'
except Exception as exc:  # report, then fail the run
    status.update(result='fail', error=f'{type(exc).__name__}: {str(exc)[-1500:]}')
print('PROBE ' + json.dumps(status), flush=True)
dist.barrier(group=control)
sys.exit(0 if status['result'] == 'pass' else 1)
