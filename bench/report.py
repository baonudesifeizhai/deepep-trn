"""Aggregate a results directory into summary.md and ops.csv.

python bench/report.py results/<run_id>
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

STAGES = ('in_d2h', 'plan', 'pack', 'h2d', 'kernel', 'd2h', 'unpack', 'out_h2d')


def load(run: Path, prefix: str) -> list[dict]:
    rows = []
    for path in sorted(run.glob(f'{prefix}_ep*.jsonl')):
        rows += [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows


def med(d):
    return d['median_us'] if d else None


def fmt(v, digits=1):
    if v is None:
        return '-'
    if isinstance(v, float):
        return f'{v:.{digits}f}'
    return str(v)


def table(header, rows):
    out = ['| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    out += ['| ' + ' | '.join(fmt(c) for c in r) + ' |' for r in rows]
    return '\n'.join(out)


def ops_section(ops):
    rows, bd_rows = [], []
    for r in sorted(ops, key=lambda r: (r['mode'], r['preset'], r['ep'], r['tokens'], r['routing'])):
        e2e, res, b = r['e2e_us'], r['resident_us'], r['bytes_per_rank']
        lat = r.get('resident_latency_us', {'dispatch': None, 'combine': None})
        d_res, c_res = med(res['dispatch']), med(res['combine'])
        total = med(e2e['dispatch']) + med(e2e['combine'])
        host = 100 * (1 - (d_res + c_res) / total) if d_res is not None and total else None
        rows.append([r['mode'], r['preset'], r['ep'], r['tokens'], r['routing'],
                     'PASS' if r['check']['passed'] else 'FAIL',
                     med(e2e['dispatch']), med(e2e['combine']), med(e2e['dispatch_combine']),
                     d_res, c_res, med(lat['dispatch']), med(lat['combine']), host,
                     r['logical_gbps'].get('resident_dispatch'), r['logical_gbps']['e2e_dispatch'],
                     b['recv_buffer_dispatch'] / 2**20, b['recv_buffer_combine'] / 2**20,
                     r['recv_load']['imbalance']])
        for op in ('dispatch', 'combine'):
            st = r['breakdown_us'][op]
            bd_rows.append([r['mode'], r['preset'], r['ep'], r['tokens'], r['routing'], op]
                           + [st.get(s) for s in STAGES])
    header = ['mode', 'preset', 'EP', 'T/rank', 'routing', 'check', 'e2e dispatch us', 'e2e combine us',
              'e2e d+c us', 'kernel dispatch us', 'kernel combine us', 'kernel latency d us',
              'kernel latency c us', 'host share %',
              'kernel GB/s (logical, d)', 'e2e GB/s (logical, d)', 'recv buf d MiB', 'recv buf c MiB',
              'load imbalance']
    return (table(header, rows),
            table(['mode', 'preset', 'EP', 'T/rank', 'routing', 'op'] + [f'{s} us' for s in STAGES], bd_rows))


def transport_section(tr):
    rows = []
    for r in sorted(tr, key=lambda r: (r['ep'], r['width'], r['rows_per_peer'], -r['fill'])):
        rows.append([r['ep'], r['width'], r['rows_per_peer'], f"{r['fill']:g}", med(r['resident_us']),
                     med(r.get('resident_latency_us')),
                     med(r.get('with_host_copies_us')), r['payload_gbps'], r['off_rank_gbps'],
                     r['recv_buffer_bytes'] / 2**20])
    return table(['EP', 'row width', 'rows/peer', 'fill', 'kernel us', 'kernel latency us', 'with host copies us',
                  'payload GB/s', 'off-rank GB/s', 'recv buf MiB'], rows)


def baseline_section(bl, ops):
    def pair(a, b):
        return f'{fmt(a)} / {fmt(b)}'

    kernel = {}
    for r in ops:
        lat = r.get('resident_latency_us')
        if r['mode'] == 'normal' and r['routing'] == 'random' and r['resident_us']['dispatch'] and lat:
            kernel[(r['preset'], r['ep'], r['tokens'])] = pair(
                med(r['resident_us']['dispatch']) + med(r['resident_us']['combine']),
                med(lat['dispatch']) + med(lat['combine']))
    by_case = {}
    for r in bl:
        key = (r['preset'], r['ep'], r['tokens'])
        by_case.setdefault(key, {})[r['op']] = (pair(med(r['resident_us']), med(r.get('resident_latency_us')))
                                               if 'resident_us' in r else 'n/a')
    names = ['all_reduce', 'all_gather', 'reduce_scatter', 'ag_rs', 'all_to_all_single']
    rows = [[*k] + [v.get(n) for n in names] + [kernel.get(k)] for k, v in sorted(by_case.items())]
    return ('Each cell: pipelined us / latency us (sync after every call).\n\n'
            + table(['preset', 'EP', 'T/rank'] + names + ['deep_ep_trn normal d+c'], rows))


def main():
    run = Path(sys.argv[1])
    ops, tr, bl = load(run, 'ops'), load(run, 'transport'), load(run, 'baselines')
    env = json.loads((run / 'env.json').read_text()) if (run / 'env.json').exists() else {}
    parts = [f'# deep_ep_trn benchmark: {run.name}', '',
             f"- hardware: {env.get('hardware')}", f"- versions: {env.get('versions')}",
             f"- deep_ep_trn git: {env.get('deep_ep_trn_git')}", '',
             'Timing: synchronized host wall-clock, max across ranks per round, median of rounds. '
             '"kernel" = device-resident replay of the exact exchange (no host packing/copies). '
             'Logical bytes follow deep_ep (normal: received rows x H x 2; LL: selected experts x H x 2).', '']
    if ops:
        summary, bd = ops_section(ops)
        parts += ['## Ops', '', summary, '', '## Stage breakdown (max across ranks)', '', bd, '']
    if tr:
        parts += ['## Transport kernel curve', '', transport_section(tr), '']
    if bl:
        parts += ['## Baselines (device-resident collectives)', '', baseline_section(bl, ops), '']
    (run / 'summary.md').write_text('\n'.join(parts))
    if ops:
        with open(run / 'ops.csv', 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['mode', 'preset', 'ep', 'tokens', 'routing', 'check', 'e2e_dispatch_us', 'e2e_combine_us',
                        'kernel_dispatch_us', 'kernel_combine_us', 'logical_dispatch_bytes', 'imbalance']
                       + [f'{op}_{s}_us' for op in ('dispatch', 'combine') for s in STAGES])
            for r in ops:
                w.writerow([r['mode'], r['preset'], r['ep'], r['tokens'], r['routing'], r['check']['passed'],
                            med(r['e2e_us']['dispatch']), med(r['e2e_us']['combine']),
                            med(r['resident_us']['dispatch']), med(r['resident_us']['combine']),
                            r['bytes_per_rank']['logical_dispatch'], r['recv_load']['imbalance']]
                           + [r['breakdown_us'][op].get(s) for op in ('dispatch', 'combine') for s in STAGES])
    print(f'wrote {run / "summary.md"}')


if __name__ == '__main__':
    main()
