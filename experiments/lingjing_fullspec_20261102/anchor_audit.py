"""Train-only anchor staleness, exact fallback shares and actual64-window past-gap checks."""
import argparse
from pathlib import Path

import numpy as np

from common import B, read, write, sha


def main(output):
    from as_jepa_tools.benchmark_gap import GapBenchmarkWindows, staleness_table
    rows = {}
    label_specs = dict(tworoom=dict(agent=('pos_agent', 'position', 8.)),
                       cube=dict(block=('privileged_block_0_pos', 'position', .03),
                                 yaw=('privileged_block_0_yaw', 'yaw90', 15.),
                                 effector=('proprio_effector_pos', 'position', .03)))
    for task, labels in label_specs.items():
        root = B / 'data/hardtasks-v1' / (task + '-metadata')
        table = staleness_table(root, labels, windows=20000, seed=0)
        exact = {str(gap): GapBenchmarkWindows(root, gap=gap).fallback_share() for gap in [0, 1, 2, 4, 8, 16]}
        chosen = max(gap for gap in [1, 2, 4, 8, 16] if exact[str(gap)] <= .2)
        dataset = GapBenchmarkWindows(root, gap=chosen)
        picks = np.random.default_rng(1042).choice(len(dataset), 64, replace=False)
        actual = []
        for index in picks:
            item = dataset[int(index)]
            valid = item['anchor_fallback'] or item['anchor_row'] <= int(item['rows'][0]) - chosen * dataset.stride
            assert valid
            actual.append(dict(index=int(index), episode=int(item['episode']), start=int(item['start']),
                               first_row=int(item['rows'][0]), anchor_row=int(item['anchor_row']), fallback=bool(item['anchor_fallback'])))
        dataset.open()
        ranges = {name: dict(minimum=np.asarray(dataset.file[key]).min(0).tolist(), maximum=np.asarray(dataset.file[key]).max(0).tolist())
                  for name, (key, _, _) in labels.items()}
        dataset.file.close()
        rows[task] = dict(table=table, exact_fallback_share=exact, selected_gap=chosen, train_windows64=actual,
                          label_ranges=ranges, dataset_card_sha256=sha(root / 'dataset_card.json'),
                          constant_anchor=dict(stale_share={key: 1. for key in labels}),
                          scope='Physical labels are used only to describe anchor staleness, never for training or gap selection. Gap is chosen solely by train fallback<=.2. Tolerance units follow the uploaded specification: TwoRoom coordinates8, Cube metres.03/yaw90degrees15.')
    write(output, dict(passed=True, tasks=rows))
    print({task: dict(gap=row['selected_gap'], exact_fallback=row['exact_fallback_share']) for task, row in rows.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--output', required=True)
    main(p.parse_args().output)
