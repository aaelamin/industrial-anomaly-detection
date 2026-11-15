"""Run the backbone study with checkpoints stored outside the runtime."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import uuid

os.environ['MPLBACKEND'] = 'Agg'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from week3 import build_model, train_epoch, evaluate, MVTecDataset, torch, np, file_hash
from scripts.week4_preflight import cache_on_disk
from run import CATEGORIES


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def identity(config):
    names = ['run.py', 'week3.py', 'scripts/week4_colab.py',
             'scripts/week4_preflight.py', 'requirements.txt', 'configs/week4.json']
    return {'config': config, 'sources': {n: file_hash(ROOT / n) for n in names},
            'packages': {n: importlib.metadata.version(n) for n in
                         ['torch', 'torchvision', 'anomalib', 'timm', 'numpy', 'scikit-learn']}}


def trainable_state(model):
    names = {n for n, p in model.named_parameters() if p.requires_grad}
    return {n: v.detach().cpu() for n, v in model.state_dict().items()
            if n in names or not n.startswith('feature_extractor.')}


def restore_model(model, state):
    expected = set(trainable_state(model))
    if set(state) != expected:
        raise ValueError('Checkpoint parameter names differ.')
    model.load_state_dict(state, strict=False)


def rng_state(generator):
    return {'generator': generator.get_state(), 'python': random.getstate(),
            'numpy': np.random.get_state()[1].tolist(),
            'numpy_pos': int(np.random.get_state()[2]),
            'numpy_gauss': [int(np.random.get_state()[3]), float(np.random.get_state()[4])],
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state, generator):
    generator.set_state(state['generator'])
    random.setstate(state['python'])
    np.random.set_state(('MT19937', np.array(state['numpy'], dtype=np.uint32),
                        state['numpy_pos'], *state['numpy_gauss']))
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state_all(state['cuda'])


def save_checkpoint(folder, payload):
    folder.mkdir(parents=True, exist_ok=True)
    name = f"epoch-{payload['epoch']:04d}-{uuid.uuid4().hex[:8]}.pt"
    target = folder / name
    temp = folder / (name + '.partial')
    torch.save(payload, temp)
    checksum = file_hash(temp)
    temp.replace(target)
    if file_hash(target) != checksum:
        raise IOError('Checkpoint verification failed.')
    write_json(target.with_suffix('.json'), {'sha256': checksum, 'epoch': payload['epoch']})
    # Keep the previous checkpoint until the replacement has been read back.
    valid = sorted(folder.glob('epoch-*.json'), key=lambda p: p.name, reverse=True)
    for receipt in valid[2:]:
        receipt.with_suffix('.pt').unlink(missing_ok=True)
        receipt.unlink()


def load_checkpoint(folder):
    receipts = sorted(folder.glob('epoch-*.json'), reverse=True)
    for receipt in receipts:
        try:
            info = json.loads(receipt.read_text())
            path = receipt.with_suffix('.pt')
            if file_hash(path) != info['sha256']:
                raise ValueError('Checksum mismatch')
            return torch.load(path, map_location='cpu', weights_only=True)
        except (OSError, ValueError, RuntimeError, EOFError) as error:
            print(f'Skipping incomplete checkpoint {receipt.name}: {error}', flush=True)
    if receipts or list(folder.glob('*.pt')) or list(folder.glob('*.partial')):
        raise RuntimeError('No readable checkpoint. Stopping instead of silently restarting.')
    return None


def state_hash(value):
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            v = v.detach().cpu().contiguous()
            h.update(str((str(v.dtype), tuple(v.shape))).encode())
            h.update(v.numpy().tobytes())
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(str(k).encode()); visit(v[k])
        elif isinstance(v, (list, tuple)):
            for item in v:
                visit(item)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def prepare(config, category, seed, data, scratch):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    model = build_model(config, 'cuda')
    train = MVTecDataset(data, category, 'train', config['size'])
    inventory = json.loads((ROOT / 'docs/DATA_CHECK.json').read_text())
    counts = next(x for x in inventory['categories'] if x['category'] == category)
    test = MVTecDataset(data, category, 'test', config['size'])
    if len(train) != counts['train'] or len(test) != counts['test']:
        raise ValueError('Dataset image counts differ from the study inventory.')
    print(f'Caching {category} features on runtime disk.', flush=True)
    features = cache_on_disk(model, train, scratch / 'features', config['micro_batch_size'])
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad],
                                 lr=config['learning_rate'], weight_decay=config['weight_decay'])
    generator = torch.Generator().manual_seed(seed)
    return model, optimizer, generator, features, test


def payload(model, optimizer, generator, epoch, history, protocol):
    return {'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
            'rng': rng_state(generator), 'epoch': epoch, 'history': history,
            'protocol': protocol}


def apply_saved(saved, model, optimizer, generator, protocol):
    if saved['protocol'] != protocol:
        raise ValueError('Checkpoint protocol differs.')
    restore_model(model, saved['model'])
    optimizer.load_state_dict(saved['optimizer'])
    restore_rng(saved['rng'], generator)


def run(args, config, protocol):
    recovery = args.mode.startswith('recovery')
    category, seed = ('bottle', 0) if recovery else (args.category, args.seed)
    folder = args.store / ('recovery' if recovery else f'runs/{category}/seed_{seed}')
    folder.mkdir(parents=True, exist_ok=True)
    if not recovery and (folder / 'complete.json').exists():
        info = json.loads((folder / 'complete.json').read_text())
        if info['protocol'] != protocol:
            raise ValueError('Completed run protocol differs.')
        for name, sha in info['files'].items():
            if file_hash(folder / name) != sha:
                raise ValueError(f'Completed result is missing or changed: {name}')
        print(f'Already complete: {category} seed {seed}', flush=True)
        return
    if recovery and args.mode == 'recovery-save' and (folder / 'expected.json').exists():
        print('Recovery checkpoint already prepared. Use recovery-check in a fresh runtime.', flush=True)
        return
    args.scratch.mkdir(parents=True, exist_ok=True)
    required_gib = 2 if args.mode == 'recovery-check' else 4
    if shutil.disk_usage(args.scratch).free < required_gib * 1024**3:
        raise RuntimeError(f'Need at least {required_gib} GiB free on runtime disk.')
    # This directory contains only this invocation's disposable feature files.
    scratch = args.scratch / ('cache-' + uuid.uuid4().hex)
    scratch.mkdir()
    features = []
    try:
        model, optimizer, generator, features, test = prepare(config, category, seed, args.data, scratch)
        checkpoint_dir = folder / 'checkpoints'
        saved = load_checkpoint(checkpoint_dir)
        history, start = [], 0
        if saved:
            apply_saved(saved, model, optimizer, generator, protocol)
            history, start = saved['history'], saved['epoch']
            del saved
            print(f'Resuming {category} seed {seed} at epoch {start + 1}', flush=True)
        if args.mode == 'recovery-check':
            expected = json.loads((folder / 'expected.json').read_text())
            if start != 1 or expected['protocol'] != protocol:
                raise ValueError('Prepare the recovery checkpoint first.')
            if expected['gpu'] != torch.cuda.get_device_name(0):
                raise RuntimeError('Recovery comparison needs the same GPU type. Reconnect to that type.')
            loss = train_epoch(model, features, optimizer, generator, config, 'cuda')
            actual = state_hash({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                                 'rng': rng_state(generator), 'loss': loss})
            if actual != expected['state_hash']:
                raise RuntimeError('Resumed training differs from uninterrupted training. Do not start the study.')
            write_json(folder / 'passed.json', {'protocol': protocol, 'gpu': expected['gpu'],
                                               'continued_epoch_matches_exactly': True})
            print('PASS: resumed epoch, model, optimizer, and random states match exactly.', flush=True)
            return
        end = 2 if recovery else config['epochs']
        for epoch in range(start, end):
            tick = time.perf_counter()
            loss = train_epoch(model, features, optimizer, generator, config, 'cuda')
            torch.cuda.synchronize()
            history.append({'epoch': epoch + 1, 'loss': loss, 'seconds': time.perf_counter() - tick})
            print(f'{category} seed {seed}: epoch {epoch + 1}/{end}, loss {loss:.2f}', flush=True)
            if recovery and epoch == 1:
                expected = state_hash({'model': trainable_state(model), 'optimizer': optimizer.state_dict(),
                                       'rng': rng_state(generator), 'loss': loss})
                write_json(folder / 'expected.json', {'protocol': protocol, 'state_hash': expected,
                                                     'gpu': torch.cuda.get_device_name(0)})
                print('Recovery checkpoint saved at epoch 1; uninterrupted epoch 2 recorded. Reconnect and check.', flush=True)
            elif recovery or (epoch + 1) % 25 == 0 or epoch + 1 == end:
                save_checkpoint(checkpoint_dir, payload(model, optimizer, generator, epoch + 1, history, protocol))
                write_json(folder / 'progress.json', {'saved_epoch': epoch + 1, 'epochs': end,
                                                      'category': category, 'seed': seed})
        if recovery:
            return
        local_results = scratch / 'results'
        local_results.mkdir()
        metrics = evaluate(model, test, local_results, config['micro_batch_size'], 'cuda')
        metrics.update(category=category, seed=seed, epochs=end, protocol=protocol,
                       gpu=torch.cuda.get_device_name(0), training_seconds=sum(x['seconds'] for x in history))
        write_json(local_results / 'metrics.json', metrics)
        write_json(local_results / 'training.json', history)
        torch.save({'trainable_state': trainable_state(model), 'config': config,
                    'weights_sha256': config['weights_sha256']}, local_results / 'model.pt')
        files = {}
        for source in sorted(local_results.rglob('*')):
            if not source.is_file():
                continue
            relative = source.relative_to(local_results).as_posix()
            target = folder / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temp = target.with_name(target.name + '.partial')
            shutil.copyfile(source, temp)
            expected = file_hash(source)
            if file_hash(temp) != expected:
                raise IOError(f'Result copy failed: {relative}')
            temp.replace(target)
            files[relative] = expected
        write_json(folder / 'complete.json', {'protocol': protocol, 'files': files})
        # Final weights and outputs are verified; optimizer checkpoints are no longer needed.
        for p in checkpoint_dir.glob('epoch-*'):
            if p.is_file():
                p.unlink()
        print(f'Completed {category} seed {seed}: {metrics}', flush=True)
    finally:
        for feature in features:
            feature.values._mmap.close()
        shutil.rmtree(scratch)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['recovery-save', 'recovery-check', 'study', 'run'], required=True)
    parser.add_argument('--store', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=ROOT / 'data/mvtec_ad')
    parser.add_argument('--scratch', type=Path, default=ROOT / 'work/week4-cache')
    parser.add_argument('--category', choices=CATEGORIES)
    parser.add_argument('--seed', type=int, choices=[0, 1, 2])
    args = parser.parse_args()
    config = json.loads((ROOT / 'configs/week4.json').read_text())
    protocol_data = identity(config)
    protocol = digest(protocol_data)
    args.store.mkdir(parents=True, exist_ok=True)
    manifest = args.store / 'study.json'
    if manifest.exists():
        if json.loads(manifest.read_text()) != protocol_data:
            raise ValueError('Code, settings, or package versions changed. Use the original package to resume.')
    else:
        write_json(manifest, protocol_data)
    if args.mode in ('study', 'run'):
        receipt = json.loads((args.store / 'recovery/passed.json').read_text())
        if receipt['protocol'] != protocol or not receipt['continued_epoch_matches_exactly']:
            raise ValueError('Run the recovery test before starting the study.')
    if args.mode == 'study':
        if (args.data / '.download-in-progress').exists():
            raise RuntimeError('Dataset download is incomplete.')
        for category in CATEGORIES:
            for seed in config['seeds']:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--mode', 'run',
                                '--store', str(args.store), '--data', str(args.data),
                                '--scratch', str(args.scratch), '--category', category, '--seed', str(seed)],
                               check=True)
        print('All 45 runs have verified results in Drive.', flush=True)
    else:
        if args.mode == 'run' and (args.category is None or args.seed is None):
            parser.error('run requires --category and --seed')
        run(args, config, protocol)


if __name__ == '__main__':
    main()
