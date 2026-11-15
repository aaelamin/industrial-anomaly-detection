"""Train Anomalib FastFlow on the official MVTec AD split."""
import argparse
import csv
import gc
import importlib.metadata
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.cache/matplotlib'))
os.environ.setdefault('HF_HOME', str(ROOT / '.cache/huggingface'))
os.environ.setdefault('TORCH_HOME', str(ROOT / '.cache/torch'))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from anomalib.models.image.fastflow.torch_model import FastflowModel
from anomalib.models.image.fastflow.loss import FastflowLoss

CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()


class MVTecDataset(Dataset):
    def __init__(self, root, category, split, size):
        self.root = root / category
        self.paths = sorted((self.root / split).glob('*/*.png'))
        if not self.paths:
            raise FileNotFoundError(f'No images in {self.root / split}. Run scripts/download_data.py first.')
        if split == 'train' and any(p.parent.name != 'good' for p in self.paths):
            raise ValueError('Training split contains anomalous images.')
        self.size = size
        self.transform = transforms.Compose([
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        label = int(path.parent.name != 'good')
        mask = np.zeros((self.size, self.size), dtype=np.uint8)
        if label:
            mask_path = self.root / 'ground_truth' / path.parent.name / (path.stem + '_mask.png')
            with Image.open(mask_path) as image:
                mask = (np.asarray(image.convert('L').resize(
                    (self.size, self.size), Image.Resampling.NEAREST)) > 0).astype(np.uint8)
        return tensor, label, mask, str(path.relative_to(self.root))


def git_state():
    def git(*args):
        result = subprocess.run(['git', *args], cwd=ROOT, capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': git('rev-parse', 'HEAD'), 'changes': git('status', '--short')}


def save_preview(path, source, mask, anomaly_map):
    from matplotlib import colormaps
    with Image.open(source) as image:
        original = image.convert('RGB').resize((mask.shape[1], mask.shape[0]))
    scale = (anomaly_map - anomaly_map.min()) / max(float(np.ptp(anomaly_map)), 1e-12)
    heat = Image.fromarray((colormaps['inferno'](scale)[..., :3] * 255).astype(np.uint8))
    truth = Image.fromarray(mask * 255).convert('RGB')
    canvas = Image.new('RGB', (original.width * 3, original.height))
    for i, panel in enumerate([original, truth, Image.blend(original, heat, 0.5)]):
        canvas.paste(panel, (i * original.width, 0))
    canvas.save(path)


def run(args, category, seed):
    if (args.data / '.download-in-progress').exists():
        raise RuntimeError('Dataset download is incomplete. Finish scripts/download_data.py first.')
    out = args.output / ('smoke' if args.smoke else 'baseline') / args.backbone / category / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    if (out / 'metrics.json').exists():
        if not args.skip_existing:
            raise FileExistsError(f'Completed run exists: {out}. Use --skip-existing or a new --output directory.')
        previous = json.loads((out / 'config.json').read_text())
        for key in ['size', 'batch_size', 'lr', 'smoke']:
            if previous[key] != getattr(args, key):
                raise ValueError(f'Existing run has different {key}: {out}')
        if previous['epochs_run'] != (1 if args.smoke else args.epochs) or previous['data'] != str(args.data):
            raise ValueError(f'Existing run has different epochs or dataset path: {out}')
        print(f'Skipping completed run: {out}', flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    device = args.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable. Install the GPU packages or pass --device cpu.')
    train = MVTecDataset(args.data, category, 'train', args.size)
    test = MVTecDataset(args.data, category, 'test', args.size)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, num_workers=0,
                              generator=torch.Generator().manual_seed(seed))
    test_loader = DataLoader(test, batch_size=args.batch_size, num_workers=0)
    epochs = 1 if args.smoke else args.epochs
    metadata = {**vars(args), 'data': str(args.data), 'output': str(args.output),
                'category': category, 'seed': seed, 'device_used': device,
                'epochs_run': epochs, 'train_images': len(train), 'test_images': len(test),
                'git': git_state(), 'python': sys.version,
                'packages': {p: importlib.metadata.version(p) for p in
                             ['anomalib', 'torch', 'torchvision', 'timm', 'numpy', 'scikit-learn']},
                'image_score': 'maximum raw anomaly map value',
                'pixel_metric': 'pooled pixels at input resolution, nearest-neighbor mask resize',
                'checkpoint_selection': 'final epoch; no test-based selection'}
    (out / 'config.json').write_text(json.dumps(metadata, indent=2))
    start = time.perf_counter()
    model = FastflowModel((args.size, args.size), args.backbone, pre_trained=True,
                          flow_steps=8, hidden_ratio=1.0).to(device)
    metadata['pretrained_weights'] = model.feature_extractor.pretrained_cfg
    metadata['feature_layers'] = model.feature_extractor.feature_info.get_dicts()
    (out / 'config.json').write_text(json.dumps(metadata, indent=2))
    criterion = FastflowLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    with (out / 'training.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['epoch', 'loss'])
        for epoch in range(epochs):
            model.train()
            losses = []
            for step, (images, _, _, _) in enumerate(train_loader):
                optimizer.zero_grad(set_to_none=True)
                loss = criterion(*model(images.to(device)))
                if not torch.isfinite(loss):
                    raise RuntimeError('Non-finite training loss.')
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
                if args.smoke and step == 1:
                    break
            mean_loss = float(np.mean(losses))
            writer.writerow([epoch + 1, mean_loss])
            handle.flush()
            print(f'{category} seed={seed} epoch={epoch + 1}/{epochs} loss={mean_loss:.3f}', flush=True)
    torch.save(model.state_dict(), out / 'model.pt')
    model.eval()
    scores, labels, maps, masks, rows = [], [], [], [], []
    previews = set()
    (out / 'heatmaps').mkdir(exist_ok=True)
    with torch.inference_mode():
        for images, batch_labels, batch_masks, paths in test_loader:
            prediction = model(images.to(device))
            batch_maps = prediction.anomaly_map.cpu().numpy()[:, 0]
            batch_scores = prediction.pred_score.cpu().numpy().reshape(-1)
            if not np.isfinite(batch_maps).all():
                raise RuntimeError('Non-finite anomaly scores.')
            for score, label, mask, anomaly_map, path in zip(
                    batch_scores, batch_labels.numpy(), batch_masks.numpy(), batch_maps, paths):
                scores.append(float(score))
                labels.append(int(label))
                maps.append(anomaly_map.reshape(-1))
                masks.append(mask.reshape(-1))
                rows.append([path, int(label), float(score)])
                defect = Path(path).parent.name
                if defect not in previews:
                    save_preview(out / 'heatmaps' / f'{defect}.png', test.root / path, mask, anomaly_map)
                    previews.add(defect)
    with (out / 'scores.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['image', 'label', 'score'])
        writer.writerows(rows)
    metrics = {'category': category, 'seed': seed, 'smoke': args.smoke,
               'image_auroc': float(roc_auc_score(labels, scores)),
               'pixel_auroc': float(roc_auc_score(np.concatenate(masks), np.concatenate(maps))),
               'seconds': time.perf_counter() - start}
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2), flush=True)
    del model, optimizer, maps, masks
    gc.collect()
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=CATEGORIES + ['all'], default='bottle')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0])
    parser.add_argument('--data', type=Path, default=Path('data/mvtec_ad'))
    parser.add_argument('--output', type=Path, default=Path('results'))
    parser.add_argument('--backbone', choices=['resnet18', 'wide_resnet50_2'], default='resnet18')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--smoke', action='store_true', help='Two training batches, then the full category test set.')
    parser.add_argument('--skip-existing', action='store_true', help='Skip completed runs with matching settings.')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.size < 32 or args.size % 16:
        parser.error('Epochs and batch size must be positive; size must be a multiple of 16 and at least 32.')
    for category in CATEGORIES if args.category == 'all' else [args.category]:
        for seed in args.seeds:
            run(args, category, seed)


if __name__ == '__main__':
    main()
