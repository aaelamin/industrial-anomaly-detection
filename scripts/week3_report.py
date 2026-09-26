"""Check the Week 3 results and write tables and a short report."""
import argparse
import csv
import json
from pathlib import Path
import statistics

CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()


def write_csv(path, rows):
    if not rows:
        return
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def mean_std(values):
    return statistics.mean(values), statistics.stdev(values) if len(values) > 1 else None


def fmt(mean, std):
    return f'{100 * mean:.2f} ± {100 * std:.2f}' if std is not None else f'{100 * mean:.2f} (one run)'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('results/week3'))
    parser.add_argument('--output', type=Path, default=Path('reports/week3'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    completed, missing, signatures, difficult = {}, [], set(), {}
    for category in CATEGORIES:
        for seed in [0, 1, 2]:
            folder = args.root / category / f'seed_{seed}'
            path = folder / 'metrics.json'
            if not path.exists():
                missing.append(f'{category}/seed_{seed}')
                continue
            result = json.loads(path.read_text())
            config = json.loads((folder / 'config.json').read_text())
            if result['epochs'] != 500 or result['seed'] != seed or result['category'] != category:
                raise ValueError(f'Unexpected run identity or epoch count: {path}')
            signatures.add(result['protocol_hash'])
            with (folder / 'scores.csv').open() as handle:
                scores = list(csv.DictReader(handle))
            normal = [float(r['score']) for r in scores if int(r['label']) == 0]
            abnormal = [float(r['score']) for r in scores if int(r['label']) == 1]
            auc = sum((a > b) + 0.5 * (a == b) for a in abnormal for b in normal) / (len(normal) * len(abnormal))
            if abs(auc - result['image_auroc']) > 1e-10 or len(scores) != result['test_images']:
                raise ValueError(f'Score verification failed: {folder}')
            if result['protocol_hash'] != config['protocol_hash']:
                raise ValueError(f'Protocol mismatch: {folder}')
            for row in scores:
                value = float(row['score'])
                if int(row['label']):
                    error = sum((n > value) + 0.5 * (n == value) for n in normal) / len(normal)
                else:
                    error = sum((a < value) + 0.5 * (a == value) for a in abnormal) / len(abnormal)
                key = (category, row['image'], row['defect'], row['mask_fraction'])
                difficult.setdefault(key, []).append(error)
            completed[(category, seed)] = result
    if len(signatures) > 1:
        raise ValueError('Results contain different protocols. Report them separately.')
    rows, defect_rows = [], []
    for category in CATEGORIES:
        runs = [completed[(category, seed)] for seed in [0, 1, 2] if (category, seed) in completed]
        if not runs:
            continue
        row = {'category': category, 'runs': len(runs)}
        for metric in ['image_auroc', 'pixel_auroc', 'pixel_average_precision']:
            mean, std = mean_std([r[metric] for r in runs])
            row[metric + '_mean'], row[metric + '_std'] = mean, std
        row['training_hours_total'] = sum(r['training_seconds'] for r in runs) / 3600
        rows.append(row)
        for defect in runs[0]['defects']:
            mean, std = mean_std([r['defects'][defect]['image_auroc'] for r in runs])
            defect_rows.append({'category': category, 'defect': defect, 'runs': len(runs),
                                'anomalous_images': runs[0]['defects'][defect]['images'],
                                'image_auroc_mean': mean, 'image_auroc_std': std})
    failure_rows = [{'category': key[0], 'image': key[1], 'defect': key[2],
                     'mask_fraction': float(key[3]), 'runs': len(values),
                     'mean_pairwise_error': statistics.mean(values),
                     'seeds_with_ranking_error': sum(v > 0 for v in values)}
                    for key, values in difficult.items()]
    failure_rows.sort(key=lambda r: r['mean_pairwise_error'], reverse=True)
    write_csv(args.output / 'per_category.csv', rows)
    write_csv(args.output / 'per_defect.csv', defect_rows)
    write_csv(args.output / 'difficult_images.csv', failure_rows)
    status = {'complete': not missing, 'completed_runs': len(completed), 'expected_runs': 45,
              'missing': missing, 'protocol_hash': next(iter(signatures), None)}
    lines = ['# Week 3: FastFlow on MVTec AD', '',
             f"Status: {'complete' if not missing else 'incomplete'} ({len(completed)}/45 runs).", '',
             'ResNet-18; 500 epochs; effective batch size 32; seeds 0, 1 and 2. '
             'Training uses only normal images. All results use the final checkpoint.', '',
             '| Category | Runs | Image AUROC (%) | Pixel AUROC (%) | Pixel AP (%) |',
             '| --- | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [fmt(row[m + '_mean'], row[m + '_std']) for m in
                  ['image_auroc', 'pixel_auroc', 'pixel_average_precision']]
        lines.append(f"| {row['category']} | {row['runs']} | " + ' | '.join(values) + ' |')
    if not missing:
        lines += ['', '## Overall results', '']
        for metric, label, reference in [('image_auroc', 'Image AUROC', 0.979),
                                          ('pixel_auroc', 'Pixel AUROC', 0.972)]:
            means = [statistics.mean(completed[(cat, seed)][metric] for cat in CATEGORIES)
                     for seed in [0, 1, 2]]
            mean, std = mean_std(means)
            status[metric] = {'mean': mean, 'std': std, 'per_seed_macro_mean': means}
            lines.append(f'- {label}: {fmt(mean, std)}%. Paper ResNet-18 reference: '
                         f'{reference * 100:.1f}%; difference {100 * (mean - reference):+.2f} percentage points.')
        lines += ['', 'Each seed contributes one mean across all 15 categories. '
                  'The reported standard deviation is across those three means.', '',
                  '## Cases to inspect', '',
                  'The table below ranks test images by their average pairwise ranking error. '
                  'For an anomaly, this is the fraction of normal images that receive a higher score. '
                  'For a normal image, it is the fraction of anomalies that receive a lower score. '
                  'Ties count as one half. No decision threshold is chosen from the test set.', '',
                  '| Category | Image | Mean ranking error | Seeds with errors |',
                  '| --- | --- | ---: | ---: |']
        for row in failure_rows[:15]:
            lines.append(f"| {row['category']} | {row['image']} | {row['mean_pairwise_error']:.3f} | "
                         f"{row['seeds_with_ranking_error']}/3 |")
    else:
        lines += ['', 'The overall result and research interpretation are pending the remaining runs.']
    lines += ['', '## Comparison limits', '',
              'This uses Anomalib and the older reference ImageNet weights, not the authors’ original code. '
              'Pixel metrics use 256 x 256 binary masks. The paper’s 99.4% headline and main per-category '
              'table use another backbone and are not matched targets for this study. '
              'The reference implementation uses no augmentation; this study follows that choice.', '',
              'Defect-level tables compare each defect type with the normal images in its category. '
              'Small test groups can produce unstable estimates. Test-set failure analysis is descriptive; '
              'it does not establish a cause or justify tuning on these examples.', '',
              'Source: [FastFlow](https://arxiv.org/html/2111.07677v2), Tables 1 and 5. '
              'See `docs/WEEK3_PROTOCOL.md` for the fixed settings and implementation choices.', '']
    (args.output / 'week3.md').write_text('\n'.join(lines), encoding='utf-8')
    (args.output / 'status.json').write_text(json.dumps(status, indent=2))
    print(f'Checked {len(completed)}/45 runs; report saved to {args.output}', flush=True)


if __name__ == '__main__':
    main()
