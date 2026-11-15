"""Run the fixed Week 3 experiment. Resume unfinished runs from the last checkpoint."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import time

from run import (ROOT, CATEGORIES, MVTecDataset, FastflowModel, FastflowLoss,
                 git_state, save_preview, torch, np, DataLoader)
from sklearn.metrics import roc_auc_score, average_precision_score
import importlib.metadata


def file_hash(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def atomic_json(path, data):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(path)


def build_model(config, device):
    model = FastflowModel((config['size'], config['size']), config['backbone'],
                          pre_trained=False, flow_steps=config['flow_steps'],
                          hidden_ratio=config['hidden_ratio'],
                          conv3x3_only=config['conv3x3_only'])
    weights_path = Path(torch.hub.get_dir()) / 'checkpoints' / Path(config['weights_url']).name
    if not weights_path.exists():
        weights_path.parent.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(config['weights_url'], str(weights_path),
                                      hash_prefix=config['weights_sha256'])
    if file_hash(weights_path) != config['weights_sha256']:
        raise ValueError('Pretrained weight checksum does not match the fixed protocol.')
    # This verified PyTorch release uses the older tar serialization format.
    weights = torch.load(weights_path, map_location='cpu', weights_only=False)
    expected = model.feature_extractor.state_dict()
    selected = {key: weights[key] if key in weights else value for key, value in expected.items()
                if key in weights or key.endswith('num_batches_tracked')}
    model.feature_extractor.load_state_dict(selected, strict=True)
    model.feature_extractor.eval()
    return model.to(device)


def cache_features(model, dataset, batch_size, device):
    parts = [[], [], []]
    with torch.no_grad():
        for images, *_ in DataLoader(dataset, batch_size=batch_size, num_workers=0):
            for i, feature in enumerate(model.feature_extractor(images.to(device))):
                parts[i].append(feature.cpu())
    return [torch.cat(part) for part in parts]


def feature_loss(model, features):
    hidden, jacobians = [], []
    for norm, flow, feature in zip(model.norms, model.fast_flow_blocks, features):
        z, jacobian = flow(norm(feature))
        hidden.append(z)
        jacobians.append(jacobian)
    return FastflowLoss()(hidden, jacobians)


def train_epoch(model, features, optimizer, generator, config, device):
    model.train()
    order = torch.randperm(len(features[0]), generator=generator)
    batch_size, micro = config['batch_size'], config['micro_batch_size']
    usable = len(order) // batch_size * batch_size
    losses = []
    for offset in range(0, usable, batch_size):
        batch = order[offset:offset + batch_size]
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for indices in batch.split(micro):
            selected = [feature[indices].to(device) for feature in features]
            loss = feature_loss(model, selected)
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite loss; stopping this run.')
            weighted = loss * (len(indices) / batch_size)
            weighted.backward()
            total += float(weighted.detach())
        optimizer.step()
        losses.append(total)
    if not losses:
        raise ValueError('Training set is smaller than the effective batch size.')
    return float(np.mean(losses))


def evaluate(model, dataset, out, batch_size, device):
    model.eval()
    rows, all_maps, all_masks = [], [], []
    started = time.perf_counter()
    with torch.inference_mode():
        for images, labels, masks, paths in DataLoader(dataset, batch_size=batch_size):
            predictions = model(images.to(device))
            maps = predictions.anomaly_map.cpu().numpy()[:, 0]
            for anomaly_map, label, mask, path in zip(maps, labels.numpy(), masks.numpy(), paths):
                if not np.isfinite(anomaly_map).all():
                    raise RuntimeError('Non-finite anomaly map.')
                rows.append({'image': path, 'label': int(label), 'score': float(anomaly_map.max()),
                             'defect': Path(path).parent.name, 'mask_fraction': float(mask.mean())})
                all_maps.append(anomaly_map)
                all_masks.append(mask)
    inference_seconds = time.perf_counter() - started
    labels = np.array([row['label'] for row in rows])
    scores = np.array([row['score'] for row in rows])
    maps, masks = np.stack(all_maps), np.stack(all_masks)
    with (out / 'scores.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / 'maps.npz', maps=maps)
    defects = {}
    for defect in sorted({row['defect'] for row in rows} - {'good'}):
        selected = np.array([row['defect'] in ('good', defect) for row in rows])
        defects[defect] = {'images': sum(row['defect'] == defect for row in rows),
                           'image_auroc': float(roc_auc_score(labels[selected], scores[selected]))}
    previews = out / 'heatmaps'
    previews.mkdir(exist_ok=True)
    # Save the hardest-ranked normal and defective images, plus one per defect type.
    chosen = set(np.flatnonzero(labels == 0)[np.argsort(scores[labels == 0])[-3:]].tolist())
    chosen.update(np.flatnonzero(labels == 1)[np.argsort(scores[labels == 1])[:3]].tolist())
    for defect in sorted({row['defect'] for row in rows}):
        chosen.add(next(i for i, row in enumerate(rows) if row['defect'] == defect))
    for i in sorted(chosen):
        row = rows[i]
        name = f"{row['defect']}_{Path(row['image']).stem}.png"
        save_preview(previews / name, dataset.root / row['image'], masks[i], maps[i])
    return {'image_auroc': float(roc_auc_score(labels, scores)),
            'pixel_auroc': float(roc_auc_score(masks.ravel(), maps.ravel())),
            'pixel_average_precision': float(average_precision_score(masks.ravel(), maps.ravel())),
            'test_images': len(rows), 'inference_seconds': inference_seconds, 'defects': defects}


def run_one(config, category, seed, data, output, device):
    protocol_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    out = output / category / f'seed_{seed}'
    out.mkdir(parents=True, exist_ok=True)
    metadata_path = out / 'config.json'
    if metadata_path.exists():
        old = json.loads(metadata_path.read_text())
        if old['protocol_hash'] != protocol_hash:
            raise ValueError(f'Configuration differs from the existing run: {out}')
        if old['runner_sha256'] != file_hash(__file__) or old['loader_sha256'] != file_hash(ROOT / 'run.py'):
            raise ValueError(f'Source code differs from the existing run: {out}')
        if any(importlib.metadata.version(name) != version for name, version in old['packages'].items()):
            raise ValueError(f'Package versions differ from the existing run: {out}')
    if (out / 'metrics.json').exists():
        print(f'Already complete: {category}, seed {seed}', flush=True)
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    generator = torch.Generator().manual_seed(seed)
    train = MVTecDataset(data, category, 'train', config['size'])
    test = MVTecDataset(data, category, 'test', config['size'])
    model = build_model(config, device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    metadata = {'protocol': config, 'protocol_hash': protocol_hash, 'category': category, 'seed': seed,
                'train_images': len(train), 'test_images': len(test), 'device': device,
                'gpu': torch.cuda.get_device_name(0) if device == 'cuda' else None,
                'git': git_state(), 'runner_sha256': file_hash(__file__),
                'loader_sha256': file_hash(ROOT / 'run.py'),
                'packages': {name: importlib.metadata.version(name) for name in
                             ['torch', 'torchvision', 'anomalib', 'timm', 'numpy', 'scikit-learn']},
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
    atomic_json(metadata_path, metadata)
    torch.cuda.reset_peak_memory_stats() if device == 'cuda' else None
    features = cache_features(model, train, config['micro_batch_size'], device)
    checkpoint = out / 'checkpoint.pt'
    history, start_epoch, elapsed = [], 0, 0.0
    if checkpoint.exists():
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        generator.set_state(saved['generator'])
        history, start_epoch, elapsed = saved['history'], saved['epoch'], saved['training_seconds']
        print(f'Resuming {category}, seed {seed}, epoch {start_epoch}', flush=True)
    started = time.perf_counter()
    for epoch in range(start_epoch, config['epochs']):
        epoch_start = time.perf_counter()
        loss = train_epoch(model, features, optimizer, generator, config, device)
        history.append({'epoch': epoch + 1, 'loss': loss, 'seconds': time.perf_counter() - epoch_start})
        training_seconds = elapsed + time.perf_counter() - started
        atomic_json(out / 'progress.json', {'category': category, 'seed': seed, 'epoch': epoch + 1,
                                           'epochs': config['epochs'], 'loss': loss,
                                           'training_seconds': training_seconds})
        if (epoch + 1) % 25 == 0 or epoch + 1 == config['epochs']:
            temporary = out / 'checkpoint.tmp'
            torch.save({'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                        'generator': generator.get_state(), 'epoch': epoch + 1,
                        'history': history, 'training_seconds': training_seconds}, temporary)
            temporary.replace(checkpoint)
        if epoch == start_epoch or (epoch + 1) % 10 == 0:
            print(f'{category} seed={seed} epoch={epoch + 1}/{config["epochs"]} '
                  f'loss={loss:.1f} seconds={history[-1]["seconds"]:.2f}', flush=True)
    with (out / 'training.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['epoch', 'loss', 'seconds'])
        writer.writeheader()
        writer.writerows(history)
    del features
    training_seconds = elapsed + time.perf_counter() - started
    metrics = evaluate(model, test, out, config['micro_batch_size'], device)
    metrics.update({'category': category, 'seed': seed, 'epochs': config['epochs'],
                    'protocol_hash': protocol_hash, 'training_seconds': training_seconds,
                    'gpu_peak_mib': torch.cuda.max_memory_allocated() / 1024**2 if device == 'cuda' else None})
    torch.save(model.state_dict(), out / 'model.pt')
    atomic_json(out / 'metrics.json', metrics)
    print(f'Completed {category}, seed {seed}: image AUROC={metrics["image_auroc"]:.6f}, '
          f'pixel AUROC={metrics["pixel_auroc"]:.6f}', flush=True)
    del optimizer, model
    if device == 'cuda':
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--category', choices=CATEGORIES + ['all'], default='all')
    parser.add_argument('--seeds', type=int, nargs='+')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--data', type=Path, default=Path('data/mvtec_ad'))
    parser.add_argument('--output', type=Path, default=Path('results/week3'))
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/week3.json').read_text())
    if args.epochs:
        config['epochs'] = args.epochs
    if (args.data / '.download-in-progress').exists():
        raise RuntimeError('Dataset download is incomplete.')
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        for category in CATEGORIES if args.category == 'all' else [args.category]:
            for seed in args.seeds if args.seeds is not None else config['seeds']:
                run_one(config, category, seed, args.data, args.output, args.device)
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
