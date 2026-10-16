"""Recheck saved Week 4 outputs and compare them with Week 3."""
import csv
import html
import json
import os
from pathlib import Path
import statistics
import sys
from datetime import datetime, timezone

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.week4_colab import identity, digest, restore_model, write_json
from week3 import build_model, file_hash, MVTecDataset, torch, np, DataLoader
from run import CATEGORIES
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score

OUT = ROOT / 'reports/week4'
STORE = ROOT / 'results/week4'
METRICS = ['image_auroc', 'pixel_auroc', 'pixel_average_precision']


def read(path):
    return json.loads(path.read_text())


def csv_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    config = read(ROOT / 'configs/week4.json')
    manifest = read(STORE / 'study.json')
    assert identity(config) == manifest, 'Training sources, settings or packages differ'
    protocol = digest(manifest)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    checks, replays, runs, baseline, defect_rows = [], [], {}, {}, []
    model = None
    for category in CATEGORIES:
        dataset = MVTecDataset(ROOT / 'data/mvtec_ad', category, 'test', 256)
        masks = []
        for path in dataset.paths:
            mask = np.zeros((256, 256), dtype=np.uint8)
            if path.parent.name != 'good':
                with Image.open(dataset.root / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')) as im:
                    mask = (np.asarray(im.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
            masks.append(mask)
        masks = np.stack(masks)
        for seed in config['seeds']:
            folder = STORE / 'runs' / category / f'seed_{seed}'
            receipt = read(folder / 'complete.json')
            assert receipt['protocol'] == protocol
            required = {'model.pt', 'maps.npz', 'metrics.json', 'training.json', 'scores.csv'}
            assert required <= receipt['files'].keys()
            for name, expected in receipt['files'].items():
                assert file_hash(folder / name) == expected, (folder, name)
            metrics = read(folder / 'metrics.json')
            assert (metrics['category'], metrics['seed'], metrics['epochs'], metrics['protocol']) == (category, seed, 500, protocol)
            history = read(folder / 'training.json')
            assert [row['epoch'] for row in history] == list(range(1, 501))
            assert all(np.isfinite(row['loss']) and row['seconds'] > 0 for row in history)
            assert abs(sum(r['seconds'] for r in history) - metrics['training_seconds']) < 1e-6
            rows = csv_rows(folder / 'scores.csv')
            old_folder = ROOT / 'results/week3' / category / f'seed_{seed}'
            old_rows = csv_rows(old_folder / 'scores.csv')
            assert len(rows) == len(old_rows) == len(dataset) == metrics['test_images']
            for row, old, path in zip(rows, old_rows, dataset.paths):
                assert Path(row['image']) == Path(old['image']) == path.relative_to(dataset.root)
                assert int(row['label']) == int(old['label']) == int(path.parent.name != 'good')
                assert row['defect'] == path.parent.name
            with np.load(folder / 'maps.npz') as saved:
                maps = saved['maps']
            assert maps.shape == masks.shape and np.isfinite(maps).all()
            labels = np.array([int(r['label']) for r in rows])
            scores = np.array([float(r['score']) for r in rows])
            np.testing.assert_array_equal(scores, maps.max(axis=(1, 2)))
            np.testing.assert_allclose([float(r['mask_fraction']) for r in rows], masks.mean(axis=(1, 2)), rtol=0, atol=1e-12)
            recalculated = dict(zip(METRICS, [roc_auc_score(labels, scores),
                roc_auc_score(masks.ravel(), maps.ravel()),
                average_precision_score(masks.ravel(), maps.ravel())]))
            for key, value in recalculated.items():
                assert abs(value - metrics[key]) < 1e-12, (folder, key)
            old_metrics = read(old_folder / 'metrics.json')
            old_scores = np.array([float(r['score']) for r in old_rows])
            assert abs(roc_auc_score(labels, old_scores) - old_metrics['image_auroc']) < 1e-12
            for defect, detail in metrics['defects'].items():
                selected = np.array([r['defect'] in ('good', defect) for r in rows])
                assert sum(r['defect'] == defect for r in rows) == detail['images']
                assert abs(roc_auc_score(labels[selected], scores[selected]) - detail['image_auroc']) < 1e-12
                defect_rows.append({'category': category, 'seed': seed, 'defect': defect,
                    'images': detail['images'], 'week3_image_auroc': old_metrics['defects'][defect]['image_auroc'],
                    'week4_image_auroc': detail['image_auroc']})
            y, x = np.unravel_index(maps.reshape(len(maps), -1).argmax(axis=1), (256, 256))
            edge = (y < 16) | (y >= 240) | (x < 16) | (x >= 240)
            inside = masks[np.arange(len(masks)), y, x].astype(bool)
            checks.append({'category': category, 'seed': seed, 'verified_files': len(receipt['files']),
                'test_images': len(rows), 'epochs': len(history), 'saved_metrics_match': True,
                'normal_peak_outer_16px_fraction': float(edge[labels == 0].mean()),
                'anomaly_peak_in_mask_fraction': float(inside[labels == 1].mean()), **recalculated})
            if seed == 0 and category in ('bottle', 'hazelnut', 'screw'):
                if model is None:
                    model = build_model(config, 'cuda')
                state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
                assert state['config'] == config and state['weights_sha256'] == config['weights_sha256']
                restore_model(model, state['trainable_state'])
                model.eval()
                actual = []
                with torch.inference_mode():
                    for images, *_ in DataLoader(dataset, batch_size=config['micro_batch_size']):
                        actual.append(model(images.cuda()).anomaly_map[:, 0].cpu().numpy())
                actual = np.concatenate(actual)
                np.testing.assert_allclose(actual, maps, rtol=1e-5, atol=1e-6)
                replays.append({'category': category, 'seed': seed, 'images': len(rows),
                    'max_map_difference': float(np.abs(actual - maps).max())})
                del state, actual
            runs[category, seed], baseline[category, seed] = metrics, old_metrics
            write_json(OUT / 'checks.json', {'complete': False, 'protocol': protocol, 'runs': checks, 'replay': replays})
            print(f'Checked {len(checks)}/45: {category} seed {seed}', flush=True)
    assert len(checks) == 45 and len(replays) == 3
    render(runs, baseline, checks, replays, protocol, config)


def render(runs, baseline, checks, replays, protocol, config):
    defect_rows = [{'category': c, 'seed': s, 'defect': defect, 'images': detail['images'],
        'week3_image_auroc': baseline[c, s]['defects'][defect]['image_auroc'],
        'week4_image_auroc': detail['image_auroc']}
        for (c, s), metrics in runs.items() for defect, detail in metrics['defects'].items()]
    summary = {}
    for metric in METRICS:
        summary[metric] = {}
        for name, source in [('week3', baseline), ('week4', runs)]:
            values = [statistics.mean(source[c, s][metric] for c in CATEGORIES) for s in config['seeds']]
            summary[metric][name] = {'mean': statistics.mean(values), 'std': statistics.stdev(values), 'per_seed_macro_mean': values}
        summary[metric]['difference_pp'] = 100 * (summary[metric]['week4']['mean'] - summary[metric]['week3']['mean'])
    table = []
    for category in CATEGORIES:
        row = {'category': category}
        for metric in METRICS:
            for name, source in [('week3', baseline), ('week4', runs)]:
                values = [source[category, s][metric] for s in config['seeds']]
                row[f'{name}_{metric}_mean'] = statistics.mean(values)
                row[f'{name}_{metric}_std'] = statistics.stdev(values)
            row[f'{metric}_difference_pp'] = 100 * (row[f'week4_{metric}_mean'] - row[f'week3_{metric}_mean'])
        table.append(row)
    write_csv(OUT / 'per_category.csv', table)
    write_csv(OUT / 'per_defect.csv', defect_rows)
    write_json(OUT / 'summary.json', summary)
    write_json(OUT / 'checks.json', {'complete': True, 'checked_utc': datetime.now(timezone.utc).isoformat(),
        'protocol': protocol, 'runs': checks, 'replay': replays,
        'scope': 'All Week 4 file receipts, epoch histories, image/pixel metrics and defect scores. Same test paths and labels as Week 3. Three Week 4 models replayed. Week 3 pixel checks are from its earlier audit.'})
    sections = []
    count = 0
    for category in CATEGORIES:
        items = []
        for seed in config['seeds']:
            for path in sorted((STORE / 'runs' / category / f'seed_{seed}' / 'heatmaps').glob('*.png')):
                count += 1
                url = path.relative_to(STORE).as_posix()
                items.append(f'<figure><figcaption>Seed {seed}: {html.escape(path.stem)}</figcaption><img loading="lazy" src="{url}" alt="Input, defect mask and prediction for {html.escape(path.stem)}"></figure>')
        sections.append(f'<details><summary>{category}</summary>{"".join(items)}</details>')
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Week 4 heatmaps</title><style>body{font:16px system-ui;max-width:1050px;margin:32px auto;padding:0 20px;background:#fafafa;color:#222}summary{cursor:pointer;font-size:22px;padding:12px}img{width:100%;max-width:768px}figure{margin:20px 0}figcaption{margin-bottom:6px}</style><h1>Week 4 heatmaps</h1>'
    page += f'<p>{count} saved previews across 15 categories and three seeds. Left: input. Middle: supplied defect mask. Right: model heatmap overlay.</p><p>Good means a normal test image. These previews include difficult examples and one of each defect type, not every test image. Each heatmap has its own colour scale, so brightness is not comparable between images.</p>' + ''.join(sections) + '</html>'
    (STORE / 'heatmaps.html').write_text(page, encoding='utf-8')
    lines = ['# Week 4: FastFlow with WideResNet-50', '',
        'We repeated the MVTec AD experiment with WideResNet-50-2 as the feature extractor. All 15 categories used seeds 0, 1 and 2, with 500 epochs per run. Training used normal images only. Evaluation used the final checkpoint and the same test images as Week 3.', '',
        '## What changed', '',
        'Week 3 used ResNet-18. This experiment used WideResNet-50-2 and its reference pretrained weights. It also used alternating 3x3 and 1x1 flow convolutions instead of all 3x3. The microbatch changed from 8 to 4 to fit memory, while the effective batch stayed at 32. Both experiments used 256 x 256 images, eight flow steps, the same learning rate and no augmentation.', '',
        'This compares two FastFlow configurations. It does not isolate the effect of backbone size, because the pretrained weights and flow convolutions also changed.', '',
        '## Results', '', '| Metric | Week 3, ResNet-18 | Week 4, WideResNet-50 | Change (points) |', '| --- | ---: | ---: | ---: |']
    for metric, label in zip(METRICS, ['Image AUROC', 'Pixel AUROC', 'Pixel AP']):
        values = summary[metric]
        cells = [f"{100*values[w]['mean']:.2f} ± {100*values[w]['std']:.2f}%" for w in ('week3', 'week4')]
        lines.append(f"| {label} | {' | '.join(cells)} | {values['difference_pp']:+.2f} |")
    lines += ['', 'Each seed contributes an average across the 15 categories. The table shows the mean and sample standard deviation of those three averages. These are ranking metrics, not percentages of images classified correctly.', '',
        '| Category | Week 3 image AUROC | Week 4 image AUROC | Change (points) | Week 3 pixel AUROC | Week 4 pixel AUROC |', '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in table:
        lines.append(f"| {row['category']} | {100*row['week3_image_auroc_mean']:.2f}% | {100*row['week4_image_auroc_mean']:.2f}% | {row['image_auroc_difference_pp']:+.2f} | {100*row['week3_pixel_auroc_mean']:.2f}% | {100*row['week4_pixel_auroc_mean']:.2f}% |")
    lines += ['', '## What happened to screw and hazelnut?', '',
        'Both improved on average, but neither caught up with the stronger categories. Hazelnut improved in all three corresponding seeds. Screw improved in seeds 1 and 2, while seed 0 got worse. The larger configuration helped, but it did not give a consistent fix for screw.', '',
        '| Category | Seed | Week 3 image AUROC | Week 4 image AUROC | Week 4 pixel AP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for category in ('hazelnut', 'screw'):
        for seed in config['seeds']:
            old, new = baseline[category, seed], runs[category, seed]
            lines.append(f"| {category} | {seed} | {100*old['image_auroc']:.2f}% | {100*new['image_auroc']:.2f}% | {100*new['pixel_average_precision']:.2f}% |")
    lines += ['', 'Screw still has low pixel AP despite its high pixel AUROC. That means we should not describe its heatmaps as precise defect masks. These metrics measure different things, and the defect covers only a small part of many screw images.', '',
        'We repeated the Week 3 check of where each map reaches its highest value. These are averages across the three seeds.', '',
        '| Category | Week 3 normal peaks at border | Week 4 normal peaks at border | Week 3 defect peaks inside mask | Week 4 defect peaks inside mask |',
        '| --- | ---: | ---: | ---: | ---: |']
    previous_checks = read(ROOT / 'reports/week3/audit/checks.json')['runs']
    for category in ('bottle', 'hazelnut', 'screw'):
        values = [100 * statistics.mean(r[key] for r in group if r['category'] == category)
            for key in ('normal_peak_outer_16px_fraction', 'anomaly_peak_in_mask_fraction')
            for group in (previous_checks, checks)]
        lines.append(f"| {category} | " + ' | '.join(f'{v:.2f}%' for v in values) + ' |')
    lines += ['', 'Border means the outer 16 pixels of the 256 x 256 map, as in the Week 3 check. It is only a rough location check, not a background segmentation or proof of the cause.', '',
        '## What to do next', '',
        'The [follow-up comparison](followup.md) examines the loss curves and matched images. It found a late loss spike in pill seed 0 and persistent background responses in selected screw and hazelnut images. The [Week 5 plan](../../docs/WEEK5_PLAN.md) proposes inspecting the individual feature levels before changing their fusion.', '',
        'Keep all three seeds and use these results for the Week 4 comparison. Carpet and pill also dropped on average, with particularly weak seed-0 runs. We should inspect those alongside screw and hazelnut rather than rerun only the weak results until they improve.', '',
        'Week 5 is the multi-scale flow implementation. FastFlow already uses three feature levels, so the group needs to choose a specific change to how those levels are combined. Keep one configuration fixed and compare the existing map averaging with that change. Decide the settings before running, use normal training data for fitting, and avoid choosing settings from the test scores we just inspected.', '',
        'If we want to claim that backbone size alone caused an improvement, we first need a comparison with the same flow convolution pattern. The experiment here supports a comparison of configurations, not that narrower claim. Reproducing the paper remains a separate unresolved issue.', '',
        '## Checking the saved runs', '',
        'All 45 runs passed the file checksum checks and had 500 recorded epochs with finite losses. Image scores matched the maximum of each saved anomaly map. Image AUROC, pixel AUROC, pixel AP and defect-level image AUROC were recalculated against the test labels and masks and matched the saved metrics within 1e-12. Test paths and labels matched Week 3.', '',
        'The bottle, hazelnut and screw models from seed 0 were loaded again and run on their full test sets. Their maps matched the saved outputs within the replay tolerance (rtol 1e-5, atol 1e-6). Exact differences are in [checks.json](checks.json). This checks result integrity, not whether our implementation reproduces the paper.', '',
        '## Files', '', '- [Category comparison and standard deviations](per_category.csv)', '- [Defect scores for each seed](per_defect.csv)', '- [Overall numbers](summary.json)', '- [Checks and model replay](checks.json)',
        '- Local gallery: `results/week4/heatmaps.html`. Open it in a browser. It needs the heatmap folders beside it, which are excluded from Git.',
        '- Full outputs: `results/week4/runs/<category>/seed_<n>/`.', '',
        'To repeat these checks and rebuild the tables and gallery, run `python scripts/week4_report.py` from the repository folder. This needs both studies, the dataset and a CUDA GPU. It runs inference for three saved models and does not train.', '']
    (OUT / 'week4.md').write_text('\n'.join(lines), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Finished. Gallery contains {count} previews.', flush=True)


if __name__ == '__main__':
    if sys.argv[1:] == ['--render-only']:
        audit = read(OUT / 'checks.json')
        assert audit['complete'] and len(audit['runs']) == 45 and len(audit['replay']) == 3
        config = read(STORE / 'study.json')['config']
        runs = {(c, s): read(STORE / 'runs' / c / f'seed_{s}' / 'metrics.json') for c in CATEGORIES for s in config['seeds']}
        baseline = {(c, s): read(ROOT / 'results/week3' / c / f'seed_{s}' / 'metrics.json') for c in CATEGORIES for s in config['seeds']}
        render(runs, baseline, audit['runs'], audit['replay'], audit['protocol'], config)
    else:
        main()
