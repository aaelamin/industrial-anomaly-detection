"""Compare training histories and matched test images without training."""
import csv
import json
import os
from pathlib import Path

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR'] = str(ROOT / '.cache/matplotlib')
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

OUT = ROOT / 'reports/week4/followup'
GALLERY = ROOT / 'results/week4/followup'
CATEGORIES = ['bottle', 'hazelnut', 'screw', 'carpet', 'pill']


def read_rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def folder(week, category, seed):
    return ROOT / f'results/week{week}' / ('runs' if week == 4 else '') / category / f'seed_{seed}'


def write_csv(path, rows):
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    GALLERY.mkdir(parents=True, exist_ok=True)
    histories, images, selected, defects = [], [], [], []
    fig, axes = plt.subplots(len(CATEGORIES), 2, figsize=(12, 15), constrained_layout=True)
    tailfig, tailaxes = plt.subplots(len(CATEGORIES), 2, figsize=(12, 15), constrained_layout=True)
    for ci, category in enumerate(CATEGORIES):
        scores_by_run, maps_by_run = {}, {}
        for week in (3, 4):
            ax = axes[ci, week - 3]
            tailax = tailaxes[ci, week - 3]
            # The loss sums over latent elements. Divide by their fixed count for plotting.
            dimensions = (64*64*64 + 128*32*32 + 256*16*16) * (4 if week == 4 else 1)
            for seed in range(3):
                run = folder(week, category, seed)
                history = read_rows(run / 'training.csv') if week == 3 else json.loads((run / 'training.json').read_text())
                loss = np.array([float(r['loss']) for r in history])
                epochs = np.array([int(r['epoch']) for r in history])
                assert len(loss) == 500 and np.isfinite(loss).all()
                assert np.array_equal(epochs, np.arange(1, 501))
                metrics = json.loads((run / 'metrics.json').read_text())
                ax.plot(epochs, loss / dimensions, label=f'Seed {seed}', linewidth=1, alpha=.85)
                tailax.plot(epochs[-100:], loss[-100:] / dimensions, label=f'Seed {seed}', linewidth=1)
                late = loss[-100:] / dimensions
                jumps = np.diff(loss / dimensions)
                index = int(np.argmax(jumps))
                histories.append({'category': category, 'week': week, 'seed': seed,
                    'image_auroc': metrics['image_auroc'], 'pixel_ap': metrics['pixel_average_precision'],
                    'first_loss_per_element': float(loss[0]/dimensions),
                    'last50_mean_loss_per_element': float(loss[-50:].mean()/dimensions),
                    'last100_slope_per_epoch': float(np.polyfit(np.arange(100), late, 1)[0]),
                    'largest_upward_step_per_element': float(jumps[index]),
                    'largest_upward_step_epoch': index + 2,
                    'last100_upward_step_count': int((np.diff(late)>0).sum()),
                    'last100_std_per_element': float(late.std())})
                rows = read_rows(run / 'scores.csv')
                scores_by_run[week, seed] = rows
                with np.load(run / 'maps.npz') as saved:
                    maps_by_run[week, seed] = saved['maps']
                for defect, value in metrics['defects'].items():
                    defects.append({'category': category, 'week': week, 'seed': seed, 'defect': defect, **value})
            for plot in (ax, tailax):
                plot.set_title(f'{category}: Week {week}')
                plot.set_xlabel('Epoch')
                plot.set_ylabel('Loss / latent elements')
                if max(abs(v) for v in plot.get_ylim()) > 10:
                    plot.set_yscale('symlog', linthresh=10)
                plot.grid(alpha=.2)
                plot.legend(fontsize=8)
        reference = scores_by_run[3, 0]
        assert all([r['image'] for r in rows] == [r['image'] for r in reference] for rows in scores_by_run.values())
        labels = np.array([int(r['label']) for r in reference])
        error_by_run = {}
        for key, rows in scores_by_run.items():
            scores = np.array([float(r['score']) for r in rows])
            normals, anomalies = scores[labels == 0], scores[labels == 1]
            error = np.array([float(np.mean((anomalies < value) + .5*(anomalies == value))) if label == 0
                else float(np.mean((normals > value) + .5*(normals == value))) for value, label in zip(scores, labels)])
            error_by_run[key] = error
            metric = json.loads((folder(key[0], category, key[1]) / 'metrics.json').read_text())
            assert abs(float(error.mean()) - (1-metric['image_auroc'])) < 1e-12
        mean_error = np.mean(list(error_by_run.values()), axis=0)
        choices = []
        for label in (0, 1):
            candidates = np.flatnonzero(labels == label)
            choices.extend([(int(candidates[np.argmax(mean_error[candidates])]), 'highest mean ranking error'),
                            (int(candidates[np.argmin(mean_error[candidates])]), 'lowest mean ranking error')])
        # Keep the four Week 3 walkthrough examples so their changes can be inspected directly.
        prior = {'hazelnut': ['test/good/001.png', 'test/crack/013.png'],
                 'screw': ['test/good/014.png', 'test/scratch_neck/023.png']}.get(category, [])
        for name in prior:
            choices.append((next(i for i, row in enumerate(reference) if Path(row['image']).as_posix() == name), 'Week 3 walkthrough'))
        unique = {}
        for i, reason in choices:
            unique.setdefault(i, []).append(reason)
        for i, row in enumerate(reference):
            record = {'category': category, 'image': Path(row['image']).as_posix(), 'label': int(labels[i])}
            for (week, seed), errors in error_by_run.items():
                record[f'w{week}_s{seed}_ranking_error'] = float(errors[i])
            images.append(record)
        for i, reasons in unique.items():
            path = Path(reference[i]['image'])
            image = np.array(Image.open(ROOT / 'data/mvtec_ad' / category / path).convert('RGB').resize((256, 256)))
            mask = np.zeros((256, 256), dtype=np.uint8)
            if labels[i]:
                mask_path = ROOT / 'data/mvtec_ad' / category / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')
                mask = np.array(Image.open(mask_path).convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0
            f, axs = plt.subplots(2, 4, figsize=(12, 6.5), constrained_layout=True)
            axs[0, 0].imshow(image)
            axs[0, 0].set_title('Input')
            axs[1, 0].imshow(mask, cmap='gray', vmin=0, vmax=1)
            axs[1, 0].set_title('Dataset defect mask')
            for week in (3, 4):
                for seed in range(3):
                    ax = axs[week-3, seed+1]
                    values = maps_by_run[week, seed][i]
                    scaled = (values-values.min()) / max(float(np.ptp(values)), 1e-12)
                    overlay = .5*(image/255) + .5*plt.colormaps['inferno'](scaled)[..., :3]
                    ax.imshow(overlay)
                    y, x = np.unravel_index(values.argmax(), values.shape)
                    ax.plot(x, y, '+', color='cyan', markersize=12, markeredgewidth=1.5)
                    if mask.any():
                        ax.contour(mask, levels=[.5], colors=['lime'], linewidths=.6)
                    error = error_by_run[week, seed][i]
                    ax.set_title(f'Week {week}, seed {seed}\nRanking error {100*error:.1f}%')
            for ax in axs.ravel():
                ax.axis('off')
            f.suptitle(f'{category} / {path.as_posix()}\nEach map scaled separately. Cyan +: score peak. Green: defect boundary.', fontsize=11)
            name = f'{category}_{path.parent.name}_{path.stem}.png'
            f.savefig(GALLERY / name, dpi=140)
            plt.close(f)
            selected.append({'category': category, 'image': path.as_posix(), 'label': int(labels[i]),
                'selection': ', '.join(reasons), 'figure': name})
        print(f'Compared {category}', flush=True)
    fig.suptitle('Loss histories for matched runs\nLower means a better training objective, not necessarily better anomaly detection.', fontsize=14)
    fig.savefig(OUT / 'loss_curves.png', dpi=150)
    plt.close(fig)
    tailfig.suptitle('Last 100 epochs\nSymmetric log axis keeps large spikes visible. Linear between -10 and +10.', fontsize=14)
    tailfig.savefig(OUT / 'loss_tail.png', dpi=150)
    plt.close(tailfig)
    write_csv(OUT / 'loss_summary.csv', histories)
    write_csv(OUT / 'image_ranking.csv', images)
    write_csv(OUT / 'selected_images.csv', selected)
    write_csv(OUT / 'defect_scores.csv', defects)
    page = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Matched Week 3 and Week 4 images</title>',
        '<style>body{font:16px system-ui;max-width:1250px;margin:30px auto;padding:20px}img{width:100%}summary{font-size:24px;cursor:pointer;margin:20px 0}</style>',
        '<h1>Same images across both configurations and all three seeds</h1>',
        '<p>Ranking error is the fraction of opposite-label images ranked on the wrong side, with ties counted as half. Lower is better. It is not a classification error rate at a chosen threshold.</p>',
        '<p>For each category we selected the hardest and easiest normal and defective image by average ranking error over all six runs, plus the earlier Week 3 examples. This is a descriptive selection from the test set, not an independent evaluation sample.</p>',
        '<p>Each map has its own colour scale. Compare the location of responses, not brightness. Cyan + marks the maximum. Green outlines the supplied defect mask. MVTec AD images are used under CC BY-NC-SA 4.0.</p>',
        '<details><summary>Training curves</summary><img src="../../../reports/week4/followup/loss_curves.png" alt="Training loss for five categories"><img src="../../../reports/week4/followup/loss_tail.png" alt="Last 100 epochs"></details>']
    for category in CATEGORIES:
        page.append(f'<details><summary>{category}</summary>')
        for item in selected:
            if item['category'] == category:
                page.append(f'<p>{item["image"]}: {item["selection"]}</p><img loading="lazy" src="{item["figure"]}" alt="Matched {category} heatmaps">')
        page.append('</details>')
    page.append('</html>')
    (GALLERY / 'index.html').write_text('\n'.join(page), encoding='utf-8')
    print(f'Saved {len(selected)} matched comparisons.', flush=True)


if __name__ == '__main__':
    main()
