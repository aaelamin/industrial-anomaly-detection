"""Compare fixed image-score summaries without training or changing saved maps."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[name] = '1'
import csv
import hashlib
import json
from pathlib import Path
import statistics
import time

import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/week4/image_scores'
METHODS = ['maximum', 'top_0.1_percent_mean', 'top_1_percent_mean', 'whole_map_mean', 'local_9x9_mean_maximum']


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    results, scores, checks, defects = [], [], [], []
    folders = sorted((ROOT / 'results/week4/runs').glob('*/seed_*'))
    assert len(folders) == 45
    for folder in folders:
        category, seed = folder.parent.name, int(folder.name.split('_')[1])
        receipt = json.loads((folder / 'complete.json').read_text())
        for name in ('maps.npz', 'scores.csv', 'metrics.json'):
            assert sha(folder / name) == receipt['files'][name], str(folder / name)
        with (folder / 'scores.csv').open(newline='') as handle:
            rows = list(csv.DictReader(handle))
        metrics = json.loads((folder / 'metrics.json').read_text())
        with np.load(folder / 'maps.npz', allow_pickle=False) as archive:
            maps = archive['maps']
        assert maps.shape == (len(rows), 256, 256) and np.isfinite(maps).all()
        paths = sorted((ROOT / 'data/mvtec_ad' / category / 'test').glob('*/*.png'))
        assert [p.relative_to(ROOT / 'data/mvtec_ad' / category).as_posix() for p in paths] == [r['image'].replace('\\', '/') for r in rows]
        labels = np.array([int(row['label']) for row in rows])
        assert all(int(row['label']) == int(row['defect'] != 'good') for row in rows)
        values = {name: [] for name in METHODS}
        for item, row in zip(maps, rows):
            flat = item.ravel()
            maximum = float(flat.max())
            assert maximum == float(row['score'])
            values['maximum'].append(maximum)
            for fraction, method in ((0.001, METHODS[1]), (0.01, METHODS[2])):
                count = int(np.ceil(len(flat) * fraction))
                top = np.partition(flat, len(flat)-count)[-count:]
                value = float(top.mean(dtype=np.float64))
                assert float(flat.min()) <= value <= maximum
                values[method].append(value)
            values['whole_map_mean'].append(float(flat.mean(dtype=np.float64)))
            # A fixed neighborhood tests whether a single isolated peak dominates.
            values['local_9x9_mean_maximum'].append(float(uniform_filter(item.astype(np.float64), size=9, mode='reflect').max()))
        for method in METHODS:
            predicted = np.array(values[method])
            auc = float(roc_auc_score(labels, predicted))
            if method == 'maximum':
                assert abs(auc - metrics['image_auroc']) < 1e-12
            results.append({'category': category, 'seed': seed, 'method': method, 'image_auroc': auc,
                            'change_points': 100*(auc-metrics['image_auroc'])})
            for row, value in zip(rows, predicted):
                scores.append({'category': category, 'seed': seed, 'image': row['image'],
                               'label': row['label'], 'defect': row['defect'], 'method': method, 'score': float(value)})
            for defect in sorted({r['defect'] for r in rows} - {'good'}):
                selected = np.array([r['defect'] in ('good', defect) for r in rows])
                defects.append({'category': category, 'seed': seed, 'defect': defect, 'method': method,
                                'image_auroc': float(roc_auc_score(labels[selected], predicted[selected]))})
        checks.append({'category': category, 'seed': seed, 'maps_sha256': receipt['files']['maps.npz'],
                       'scores_sha256': receipt['files']['scores.csv'], 'maximum_matches_saved': True})
        del maps
        print(f'{len(checks)}/45: {category} seed {seed}', flush=True)
    overall = {}
    for method in METHODS:
        means = [statistics.mean(r['image_auroc'] for r in results if r['method']==method and r['seed']==seed) for seed in (0, 1, 2)]
        overall[method] = {'mean': statistics.mean(means), 'std': statistics.stdev(means), 'seed_means': means}
    write_csv(OUT / 'per_run.csv', results)
    write_csv(OUT / 'image_scores.csv', scores)
    write_csv(OUT / 'per_defect.csv', defects)
    summary = {'complete': True, 'runs': 45, 'methods': METHODS, 'overall': overall, 'checks': checks,
               'source_sha256': sha(Path(__file__)), 'elapsed_seconds': time.perf_counter()-started,
               'exploratory': True, 'limits': 'Fixed summaries applied after inspecting the original test results. No retraining, masks, per-image normalization, per-category selection or changes to baseline metrics. Not an independent validation or proof of the paper scoring rule.'}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    lines = ['# Checking image scores from saved Week 4 maps', '',
             'We compared five fixed ways of reducing each saved map to one image score across all 15 categories and three seeds. No models were trained, and no original outputs or running-study settings were changed. This check used CPU computation only.', '',
             'Top 0.1% and top 1% average the 66 and 656 highest pixels in a 256 by 256 map. The local score takes the maximum after a fixed 9 by 9 box average with reflected edges. The whole-map mean tests a much broader summary. None uses a defect mask, per-image normalization or category-specific settings.', '',
             '| Image score | Mean image AUROC | Seed std | Change from maximum |',
             '| --- | ---: | ---: | ---: |']
    base = overall['maximum']['mean']
    for method in METHODS:
        item = overall[method]
        lines.append(f"| {method} | {100*item['mean']:.2f}% | {100*item['std']:.2f} | {100*(item['mean']-base):+.2f} points |")
    lines += ['', 'Each seed is first averaged equally across the 15 categories. The table reports the mean and sample standard deviation of those three averages.', '',
              '## Category means across three seeds', '',
              '| Category | Maximum | Top 0.1% | Top 1% | Whole map | Local 9 by 9 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for category in sorted({r['category'] for r in results}):
        means = [statistics.mean(r['image_auroc'] for r in results if r['category']==category and r['method']==method) for method in METHODS]
        lines.append('| '+category+' | '+' | '.join(f'{100*v:.2f}%' for v in means)+' |')
    lines += ['', 'All 45 saved map, score and metric file hashes passed verification. Maximum scores matched the saved scores exactly and reproduced their image AUROCs. Image order and labels were checked. Pixel metrics are unchanged because the pixel maps themselves are unchanged.', '',
              'This is an exploratory sensitivity check on an already inspected test set. Better scores here cannot be presented as an independently validated improvement or as reproduction of the authors\' image-scoring rule. We report every method and category rather than picking a different winner for each category. Any chosen change needs a fixed protocol and independent confirmation.', '',
              '- [Every run](image_scores/per_run.csv)', '- [Every defect type](image_scores/per_defect.csv)',
              '- [Image scores](image_scores/image_scores.csv)', '- [Checks and summary](image_scores/summary.json)', '']
    (OUT.parent / 'image_score_check.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == '__main__':
    main()
