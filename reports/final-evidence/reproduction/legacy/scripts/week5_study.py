"""Run the fixed Week 5 comparison, with resumable category caches."""
import argparse
import ctypes
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import week5_fast_pilot as helper
from scripts.week5_model import ProjectedFlow, Projection, seed_all, torch
from scripts.week4_colab import write_json, digest, state_hash, save_checkpoint, load_checkpoint, restore_rng
from scripts.week4_preflight import DiskFeatures
from week3 import build_model, MVTecDataset, evaluate, file_hash
from run import CATEGORIES


def read(path):
    return json.loads(path.read_text())


def atomic_model(path, model, config, protocol, projection_sha, epoch):
    temp = path.with_suffix('.partial')
    torch.save({'model': model.state_dict(), 'config': config, 'protocol': protocol,
                'projection_sha256': projection_sha, 'epoch': epoch}, temp)
    sha = file_hash(temp)
    temp.replace(path)
    assert file_hash(path) == sha
    return sha


class Evaluator(torch.nn.Module):
    def __init__(self, backbone, projection, model, arm):
        super().__init__()
        self.backbone, self.projection, self.model, self.arm = backbone, projection, model, arm

    def forward(self, images):
        maps = self.model.anomaly_map(self.projection(self.backbone(images), self.arm))
        return SimpleNamespace(anomaly_map=maps)


def verify_receipt(out, protocol):
    receipt = read(out / 'complete.json')
    assert receipt['protocol'] == protocol
    for name, sha in receipt['files'].items():
        assert file_hash(out / name) == sha, str(out / name)
    return read(out / 'metrics.json')


def cleanup_cache(scratch, work):
    scratch = scratch.resolve()
    assert scratch.parent == work.resolve()
    allowed = {'level_0.npy', 'level_1.npy', 'level_2.npy', 'middle.npy', 'fusion.npy'}
    assert all(p.is_file() and p.name in allowed for p in scratch.iterdir())
    for path in scratch.iterdir():
        path.unlink()
    scratch.rmdir()


def prepare(backbone, category, config, store, work):
    marker = store / category / 'cache.json'
    if marker.exists():
        saved = read(marker)
        scratch = Path(saved['path'])
        assert scratch.resolve().parent == work.resolve()
        receipt = read(store / category / 'projection.json')
        projection_path = store / category / 'projection.pt'
        assert file_hash(projection_path) == receipt['sha256'] == saved['projection_sha256']
        train = MVTecDataset(ROOT / 'data/mvtec_ad', category, 'train', 256)
        assert {p.relative_to(train.root).as_posix(): file_hash(p) for p in train.paths} == receipt['training_files']
        for arm, sha in saved['features'].items():
            assert file_hash(scratch / f'{arm}.npy') == sha
        projection = Projection(torch.load(projection_path, map_location='cpu', weights_only=True), config['grid']).cuda()
        return scratch, projection, receipt['sha256']
    # A crash before cache publication can leave only reproducible feature files.
    for scratch in work.glob(f'{category}-*'):
        if scratch.is_dir():
            cleanup_cache(scratch, work)
    scratch, projection, info, sha = helper.prepare_category(backbone, category, config)
    write_json(marker, {'path': str(scratch.resolve()), 'projection_sha256': sha,
                        'features': {arm: file_hash(scratch / f'{arm}.npy') for arm in config['arms']}})
    return scratch, projection, sha


def run(args):
    config = read(ROOT / 'configs/week5_study.json')
    if args.smoke:
        config = {**config, 'name': 'week5-controller-smoke', 'categories': ['bottle'],
                  'seeds': [0], 'epochs': 2, 'retain_epochs': [1, 2]}
    suffix = 'study-smoke' if args.smoke else 'study'
    store, work = ROOT / f'results/week5/{suffix}', ROOT / f'work/week5-{suffix}'
    store.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    helper.STORE, helper.WORK = store, work
    if args.verify_resume:
        helper.verify_resume(args, config)
        return
    if not args.smoke:
        smoke = read(ROOT / 'results/week5/study-smoke/summary.json')
        assert smoke['complete'] and smoke['recovery_exact'] and smoke['source_sha256'] == file_hash(__file__)
        assert read(ROOT / 'reports/week5/fast/checks.json')['passed']
    lock = (work / 'controller.lock').open('a+b')
    import msvcrt
    lock.seek(0)
    if lock.read(1) == b'':
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    ctypes.windll.kernel32.SetThreadExecutionState(0x80000001)
    try:
        metadata = helper.identity(config)
        metadata['sources'].update({n: file_hash(ROOT / n) for n in ('configs/week5_study.json', 'scripts/week5_study.py')})
        protocol = digest(metadata)
        manifest = store / 'study.json'
        if manifest.exists():
            assert read(manifest) == metadata, 'Study settings or source files changed.'
        else:
            write_json(manifest, metadata)
        inventory = read(ROOT / 'docs/DATA_CHECK.json')['categories']
        for row in inventory:
            base = ROOT / 'data/mvtec_ad' / row['category']
            assert len(list((base / 'train/good').glob('*.png'))) == row['train']
            assert len(list((base / 'test').glob('*/*.png'))) == row['test']
        seed_all(0)
        backbone = build_model(read(ROOT / config['backbone_config']), 'cuda').feature_extractor
        backbone.eval().requires_grad_(False)
        backbone_hash = state_hash(backbone.state_dict())
        total = len(config['categories']) * len(config['seeds']) * len(config['arms'])
        completed = []
        for category in config['categories']:
            pending = [(seed, arm) for seed in config['seeds'] for arm in config['arms']
                       if not (store / category / f'seed_{seed}' / arm / 'complete.json').exists()]
            if pending:
                assert shutil.disk_usage(ROOT).free > 8 * 1024**3, 'Need 8 GiB free before preparing a category.'
                write_json(work / 'status.json', {'state': 'preparing', 'category': category,
                           'completed_runs': len(completed), 'total_runs': total, 'pid': os.getpid()})
                scratch, projection, projection_sha = prepare(backbone, category, config, store, work)
            for seed in config['seeds']:
                for arm in config['arms']:
                    out = store / category / f'seed_{seed}' / arm
                    out.mkdir(parents=True, exist_ok=True)
                    if (out / 'complete.json').exists():
                        completed.append(verify_receipt(out, protocol))
                        continue
                    seed_all(seed)
                    model = ProjectedFlow(config).cuda()
                    initial = state_hash(model.state_dict())
                    opt = helper.optimizer(model, config)
                    generator = torch.Generator().manual_seed(seed)
                    features = DiskFeatures(scratch / f'{arm}.npy')
                    checkpoint = out / 'checkpoints'
                    saved = load_checkpoint(checkpoint)
                    history = []
                    if saved:
                        assert saved['protocol'] == protocol and saved['projection_sha256'] == projection_sha
                        model.load_state_dict(saved['model'], strict=True)
                        opt.load_state_dict(saved['optimizer'])
                        restore_rng(saved['rng'], generator)
                        history = saved['history']
                        del saved
                        print(f'Resuming {category} seed {seed} {arm} after epoch {len(history)}', flush=True)
                    while len(history) < config['epochs']:
                        epoch = len(history) + 1
                        write_json(work / 'status.json', {'state': 'running', 'category': category, 'seed': seed,
                                   'arm': arm, 'epoch': epoch, 'epochs': config['epochs'], 'completed_runs': len(completed),
                                   'total_runs': total, 'pid': os.getpid(), 'updated_unix': time.time()})
                        row = helper.timed_epoch(model, features, opt, generator, config, epoch)
                        history.append(row)
                        if category == 'bottle' and seed == 0 and arm == 'middle' and epoch == 2:
                            expected = out / 'recovery_expected.json'
                            write_json(expected, {'protocol': protocol, 'state_hash': helper.fingerprint(model, opt, generator),
                                       'training': {k: v for k, v in row.items() if k not in ('seconds', 'epoch')}})
                            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--verify-resume',
                                       '--checkpoint', str(checkpoint), '--features', str(scratch / f'{arm}.npy'), '--expected', str(expected)]
                            if args.smoke:
                                command.append('--smoke')
                            subprocess.run(command, cwd=ROOT, check=True)
                        pause = (work / 'PAUSE').exists() or (args.stop_after_epoch and epoch >= args.stop_after_epoch)
                        if epoch in config['retain_epochs']:
                            atomic_model(out / f'model_epoch_{epoch:04d}.pt', model, config, protocol, projection_sha, epoch)
                        if epoch == 1 or epoch % config['checkpoint_every'] == 0 or epoch == config['epochs'] or pause:
                            save_checkpoint(checkpoint, helper.payload(model, opt, generator, history, protocol, projection_sha))
                        write_json(out / 'training.json', history)
                        print(f'{category} seed {seed} {arm}: {epoch}/{config["epochs"]}, {row["seconds"]:.2f}s, loss {row["loss"]:.1f}', flush=True)
                        if pause:
                            features.values._mmap.close()
                            write_json(work / 'status.json', {'state': 'paused', 'category': category, 'seed': seed,
                                       'arm': arm, 'saved_epoch': epoch, 'completed_runs': len(completed), 'total_runs': total})
                            return
                    model.eval()
                    probe = features[torch.arange(4)].cuda()
                    with torch.inference_mode():
                        before = model.anomaly_map(probe).cpu()
                    final = out / f'model_epoch_{config["epochs"]:04d}.pt'
                    restored = ProjectedFlow(config).cuda()
                    restored.load_state_dict(torch.load(final, map_location='cpu', weights_only=True)['model'])
                    restored.eval()
                    with torch.inference_mode():
                        torch.testing.assert_close(before, restored.anomaly_map(probe).cpu(), rtol=0, atol=0)
                    del restored, probe, before
                    test = MVTecDataset(ROOT / 'data/mvtec_ad', category, 'test', 256)
                    metrics = evaluate(Evaluator(backbone, projection, model, arm), test, out, config['feature_batch_size'], 'cuda')
                    assert state_hash(backbone.state_dict()) == backbone_hash
                    metrics.update(category=category, seed=seed, arm=arm, epochs=config['epochs'], protocol=protocol,
                                   initial_state_hash=initial, checkpoint_reload_exact=True, backbone_unchanged=True)
                    write_json(out / 'metrics.json', metrics)
                    retained = [out / f'model_epoch_{e:04d}.pt' for e in config['retain_epochs']]
                    files = retained + [out / name for name in ('training.json', 'metrics.json', 'scores.csv', 'maps.npz')]
                    files += list((out / 'heatmaps').glob('*.png'))
                    write_json(out / 'complete.json', {'protocol': protocol,
                               'files': {p.relative_to(out).as_posix(): file_hash(p) for p in files}})
                    completed.append(verify_receipt(out, protocol))
                    features.values._mmap.close()
                    # Completed models are retained. Optimizer recovery files are no longer needed.
                    assert checkpoint.resolve().parent == out.resolve()
                    for path in checkpoint.iterdir():
                        assert path.is_file() and path.name.startswith('epoch-') and path.suffix in ('.pt', '.json')
                        path.unlink()
                    checkpoint.rmdir()
                    del model, opt, features
                    gc.collect()
                    torch.cuda.empty_cache()
            if pending:
                del projection
            marker = store / category / 'cache.json'
            if marker.exists():
                cached_path = Path(read(marker)['path'])
                if cached_path.exists():
                    cleanup_cache(cached_path, work)
                marker.unlink()
        for category in config['categories']:
            for seed in config['seeds']:
                assert len({r['initial_state_hash'] for r in completed if r['category']==category and r['seed']==seed}) == 1
        write_json(store / 'summary.json', {'complete': True, 'runs': completed, 'protocol': protocol,
                   'source_sha256': file_hash(__file__), 'recovery_exact': read(store / 'bottle/seed_0/middle/recovery_passed.json')['fresh_process_resume_exact']})
        write_json(work / 'status.json', {'state': 'complete', 'completed_runs': len(completed), 'total_runs': total})
    except BaseException as error:
        write_json(work / 'status.json', {'state': 'stopped', 'error': str(error), 'pid': os.getpid()})
        raise
    finally:
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--stop-after-epoch', type=int)
    parser.add_argument('--verify-resume', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--features', type=Path)
    parser.add_argument('--expected', type=Path)
    run(parser.parse_args())
