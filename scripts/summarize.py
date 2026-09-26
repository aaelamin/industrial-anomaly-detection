"""Collect completed baseline runs into per-category mean and sample std."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/baseline'))
    args = parser.parse_args()
    groups = {}
    for path in sorted(args.root.glob('*/*/seed_*/metrics.json')):
        metrics = json.loads(path.read_text())
        if metrics['smoke']:
            continue
        config = json.loads((path.parent / 'config.json').read_text())
        key = (config['backbone'], metrics['category'])
        signature = tuple(config[k] for k in ['epochs_run', 'batch_size', 'size', 'lr'])
        groups.setdefault(key, []).append((metrics, signature))
    if not groups:
        raise SystemExit('No completed baseline runs found.')
    rows = []
    for (backbone, category), runs in groups.items():
        if len({signature for _, signature in runs}) != 1:
            raise ValueError(f'Mixed settings for {backbone}/{category}; summarize separately.')
        row = {'backbone': backbone, 'category': category, 'runs': len(runs)}
        for metric in ['image_auroc', 'pixel_auroc']:
            values = [run[metric] for run, _ in runs]
            row[metric + '_mean'] = statistics.mean(values)
            row[metric + '_std'] = statistics.stdev(values) if len(values) > 1 else ''
        rows.append(row)
    target = args.root / 'summary.csv'
    with target.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved {target}. Check the runs column; the course requires at least 3 seeds.')


if __name__ == '__main__':
    main()
