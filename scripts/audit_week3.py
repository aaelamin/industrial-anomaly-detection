"""Check saved results and replay selected checkpoints without changing the study."""
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from week3 import build_model, file_hash, MVTecDataset, torch, np, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from FrEIA.modules import AllInOneBlock
from anomalib.models.image.fastflow.torch_model import subnet_conv_func


def main():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    out = ROOT / 'reports/week3/audit'
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/week3.json').read_text())
    expected_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    manifest = json.loads((ROOT / 'results/week3/source/hashes.json').read_text())
    assert all(file_hash(ROOT / name) == digest for name, digest in manifest.items())
    checks, replay, diagnostics = [], [], []
    model = build_model(config, 'cuda').eval()
    pretrained = {k: v.cpu().clone() for k, v in model.feature_extractor.state_dict().items()}
    for folder in sorted((ROOT / 'results/week3').glob('*/seed_*')):
        category, seed = folder.parent.name, int(folder.name.split('_')[1])
        meta = json.loads((folder / 'config.json').read_text())
        metrics = json.loads((folder / 'metrics.json').read_text())
        assert meta['protocol_hash'] == metrics['protocol_hash'] == expected_hash
        history = list(csv.DictReader((folder / 'training.csv').open()))
        assert [int(r['epoch']) for r in history] == list(range(1, 501))
        assert all(np.isfinite(float(r['loss'])) for r in history)
        rows = list(csv.DictReader((folder / 'scores.csv').open()))
        dataset = MVTecDataset(ROOT / 'data/mvtec_ad', category, 'test', 256)
        assert len(rows) == len(dataset) == metrics['test_images']
        masks = []
        for row, path in zip(rows, dataset.paths):
            assert Path(row['image']) == path.relative_to(dataset.root)
            assert int(row['label']) == int(path.parent.name != 'good')
            from PIL import Image
            mask = np.zeros((256, 256), dtype=np.uint8)
            if int(row['label']):
                mask_path = dataset.root / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')
                with Image.open(mask_path) as im:
                    mask = (np.asarray(im.convert('L').resize((256, 256), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
            masks.append(mask)
        masks = np.stack(masks)
        with np.load(folder / 'maps.npz') as saved:
            maps = saved['maps']
        assert maps.shape == masks.shape and np.isfinite(maps).all()
        labels = np.array([int(r['label']) for r in rows])
        scores = np.array([float(r['score']) for r in rows])
        np.testing.assert_array_equal(scores, maps.max(axis=(1, 2)))
        recalculated = {'image_auroc': roc_auc_score(labels, scores),
                        'pixel_auroc': roc_auc_score(masks.ravel(), maps.ravel()),
                        'pixel_average_precision': average_precision_score(masks.ravel(), maps.ravel())}
        for key, value in recalculated.items():
            assert abs(value - metrics[key]) < 1e-12, (folder, key)
        state = torch.load(folder / 'model.pt', map_location='cpu', weights_only=True)
        assert all(torch.equal(state['feature_extractor.' + k], v) for k, v in pretrained.items())
        y, x = np.unravel_index(maps.reshape(len(maps), -1).argmax(axis=1), (256, 256))
        edge = (y < 16) | (y >= 240) | (x < 16) | (x >= 240)
        inside = masks[np.arange(len(masks)), y, x].astype(bool)
        for i, row in enumerate(rows):
            diagnostics.append({'category': category, 'seed': seed, 'image': row['image'],
                                'label': int(labels[i]), 'peak_y': int(y[i]), 'peak_x': int(x[i]),
                                'peak_in_outer_16px': bool(edge[i]), 'peak_in_defect_mask': bool(inside[i])})
        check = {'category': category, 'seed': seed, **recalculated,
                 'normal_peak_outer_16px_fraction': float(edge[labels == 0].mean()),
                 'anomaly_peak_in_mask_fraction': float(inside[labels == 1].mean()),
                 'mean_map_image_auroc_diagnostic_only': float(roc_auc_score(labels, maps.mean(axis=(1, 2)))),
                 'backbone_unchanged': True, 'saved_metrics_match': True}
        if seed == 0 and category in ('bottle', 'hazelnut', 'screw'):
            model.load_state_dict(state, strict=True)
            model.eval()
            actual = []
            with torch.inference_mode():
                for images, *_ in DataLoader(dataset, batch_size=8):
                    actual.append(model(images.cuda()).anomaly_map[:, 0].cpu().numpy())
            actual = np.concatenate(actual)
            np.testing.assert_allclose(actual, maps, rtol=1e-5, atol=1e-6)
            replay.append({'category': category, 'seed': seed, 'test_images': len(rows),
                           'max_map_difference': float(np.abs(actual - maps).max())})
            # Compare the installed flow block with FrEIA using the same trained parameters.
            block = model.fast_flow_blocks[0].module_list[0]
            reference = AllInOneBlock([(64, 64, 64)], subnet_constructor=subnet_conv_func(3, 1.0)).cuda()
            reference.load_state_dict(block.state_dict(), strict=True)
            with torch.no_grad():
                sample = torch.randn(2, 64, 64, 64, device='cuda')
                a, aj = block((sample,))
                b, bj = reference((sample,))
                torch.testing.assert_close(a[0], b[0], rtol=1e-5, atol=1e-6)
                torch.testing.assert_close(aj, bj, rtol=1e-5, atol=1e-5)
            replay[-1]['freia_block_agrees'] = True
        checks.append(check)
        (out / 'checks.json').write_text(json.dumps({'runs': checks, 'replay': replay}, indent=2))
        print(f'Passed {len(checks)}/45: {category} seed {seed}', flush=True)
    assert len(checks) == 45
    with (out / 'peak_locations.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=diagnostics[0].keys())
        writer.writeheader()
        writer.writerows(diagnostics)
    print('Audit complete.', flush=True)


if __name__ == '__main__':
    main()
