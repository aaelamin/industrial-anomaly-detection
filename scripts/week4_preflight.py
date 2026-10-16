"""Benchmark bottle before choosing where to run the full backbone comparison."""
import argparse
import importlib.metadata
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from week3 import build_model, train_epoch, evaluate, MVTecDataset, torch, np, DataLoader


class DiskFeatures:
    def __init__(self, path):
        self.values = np.load(path, mmap_mode='r')

    def __len__(self):
        return len(self.values)

    def __getitem__(self, indices):
        return torch.from_numpy(np.array(self.values[indices.numpy()], copy=True))


def cache_on_disk(model, dataset, folder, batch_size):
    folder.mkdir(parents=True, exist_ok=True)
    arrays = []
    offset = 0
    model.feature_extractor.eval()
    with torch.inference_mode():
        for images, *_ in DataLoader(dataset, batch_size=batch_size):
            features = model.feature_extractor(images.cuda())
            if not arrays:
                arrays = [np.lib.format.open_memmap(folder / f'level_{i}.npy', mode='w+',
                          dtype=np.float32, shape=(len(dataset), *f.shape[1:]))
                          for i, f in enumerate(features)]
            for target, feature in zip(arrays, features):
                target[offset:offset + len(images)] = feature.cpu().numpy()
            offset += len(images)
    for target in arrays:
        target.flush()
    del arrays
    return [DiskFeatures(folder / f'level_{i}.npy') for i in range(3)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--micro-batch-size', type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--output', type=Path, default=ROOT / 'work/week4-preflight')
    args = parser.parse_args()
    if not 3 <= args.epochs <= 10:
        parser.error('Use 3 to 10 epochs for this preflight, not a full training run.')
    out = args.output.resolve()
    if (out / 'benchmark.json').exists():
        parser.error('Benchmark output already exists. Choose another output directory.')
    out.mkdir(parents=True, exist_ok=True)
    config = json.loads((ROOT / 'configs/week4.json').read_text())
    config['micro_batch_size'] = args.micro_batch_size
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    torch.cuda.reset_peak_memory_stats()
    import os
    if os.name == 'nt':
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        started = time.perf_counter()
        model = build_model(config, 'cuda')
        train = MVTecDataset(ROOT / 'data/mvtec_ad', 'bottle', 'train', config['size'])
        test = MVTecDataset(ROOT / 'data/mvtec_ad', 'bottle', 'test', config['size'])
        print('Model loaded; caching bottle features to disk.', flush=True)
        cache_start = time.perf_counter()
        features = cache_on_disk(model, train, out / 'features', config['micro_batch_size'])
        cache_seconds = time.perf_counter() - cache_start
        indices = torch.arange(config['micro_batch_size'])
        images = torch.stack([train[int(i)][0] for i in indices])
        with torch.no_grad():
            direct = model.feature_extractor(images.cuda())
            for reference, saved in zip(direct, features):
                torch.testing.assert_close(saved[indices], reference.cpu(), rtol=0, atol=0)
        del direct, images
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                     lr=config['learning_rate'], weight_decay=config['weight_decay'])
        generator = torch.Generator().manual_seed(0)
        history = []
        for epoch in range(args.epochs):
            torch.cuda.synchronize()
            epoch_start = time.perf_counter()
            loss = train_epoch(model, features, optimizer, generator, config, 'cuda')
            torch.cuda.synchronize()
            seconds = time.perf_counter() - epoch_start
            history.append({'epoch': epoch + 1, 'seconds': seconds, 'loss': loss})
            print(f'Bottle epoch {epoch + 1}/{args.epochs}: {seconds:.2f}s, loss {loss:.1f}', flush=True)
        save_start = time.perf_counter()
        checkpoint = out / 'checkpoint.pt'
        torch.save({'epoch': args.epochs, 'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(), 'generator': generator.get_state(),
                    'config': config, 'benchmark_only': True}, checkpoint)
        save_seconds = time.perf_counter() - save_start
        model.eval()
        probe = torch.stack([test[i][0] for i in range(config['micro_batch_size'])]).cuda()
        with torch.inference_mode():
            before = model(probe).anomaly_map.cpu()
        del optimizer
        torch.cuda.empty_cache()
        load_start = time.perf_counter()
        saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.load_state_dict(saved['model'], strict=True)
        generator.set_state(saved['generator'])
        load_seconds = time.perf_counter() - load_start
        del saved
        with torch.inference_mode():
            after = model(probe).anomaly_map.cpu()
        torch.testing.assert_close(before, after, rtol=0, atol=0)
        del probe, before, after
        print('Checkpoint reload matched. Evaluating all 83 bottle test images.', flush=True)
        metrics = evaluate(model, test, out, config['micro_batch_size'], 'cuda')
        (out / 'metrics.json').write_text(json.dumps({**metrics, 'benchmark_only': True,
                                                    'epochs': args.epochs}, indent=2))
        inventory = json.loads((ROOT / 'docs/DATA_CHECK.json').read_text())
        counts = {row['category']: row['train'] for row in inventory['categories']}
        assert len(counts) == 15 and counts['bottle'] == len(train)
        steady = statistics.median(h['seconds'] for h in history[1:])
        steps_per_epoch = len(train) // config['batch_size']
        seconds_per_step = steady / steps_per_epoch
        estimate = seconds_per_step * sum(n // config['batch_size'] for n in counts.values()) * 500 * 3
        feature_bytes = sum(p.stat().st_size for p in (out / 'features').glob('*.npy'))
        report = {'benchmark_only': True, 'epochs': args.epochs, 'category': 'bottle', 'seed': 0,
                  'python_version': sys.version,
                  'packages': {name: importlib.metadata.version(name) for name in
                               ['torch', 'torchvision', 'anomalib', 'timm', 'numpy', 'scikit-learn']},
                  'config': config, 'gpu': torch.cuda.get_device_name(0),
                  'gpu_total_mib': torch.cuda.get_device_properties(0).total_memory / 1024**2,
                  'gpu_peak_allocated_mib': torch.cuda.max_memory_allocated() / 1024**2,
                  'gpu_peak_reserved_mib': torch.cuda.max_memory_reserved() / 1024**2,
                  'history': history, 'median_epoch_seconds_after_first': steady,
                  'cache_seconds': cache_seconds, 'cache_gib': feature_bytes / 1024**3,
                  'largest_category_cache_gib': feature_bytes / len(train) * max(counts.values()) / 1024**3,
                  'checkpoint_mib': checkpoint.stat().st_size / 1024**2,
                  'checkpoint_save_seconds': save_seconds, 'checkpoint_load_seconds': load_seconds,
                  'checkpoint_reload_exact': True, 'feature_cache_exact': True,
                  'test_images': metrics['test_images'], 'inference_seconds': metrics['inference_seconds'],
                  'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
                  'estimated_bottle_500_epoch_minutes': steady * 500 / 60,
                  'estimated_45_run_training_hours': estimate / 3600,
                  'estimated_45_run_checkpoint_hours': save_seconds * 20 * 45 / 3600,
                  'elapsed_seconds': time.perf_counter() - started,
                  'estimate_note': 'Extrapolated from bottle and the saved 15-category training counts; excludes most startup and evaluation costs.'}
        (out / 'benchmark.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2), flush=True)
    except torch.OutOfMemoryError:
        (out / 'error.json').write_text(json.dumps({'error': 'CUDA out of memory',
                                                   'micro_batch_size': args.micro_batch_size}))
        raise
    finally:
        if os.name == 'nt':
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
