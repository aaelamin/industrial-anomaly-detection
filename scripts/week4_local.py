"""Run Week 4 locally, keeping temporary features separate from final results."""
import argparse
import ctypes
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT / 'results/week4'
WORK = ROOT / 'work/week4-local'
CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temp.replace(path)


def check():
    reference = json.loads((ROOT / 'work/colab-recovery-test/study.json').read_text())
    passed = json.loads((ROOT / 'work/colab-recovery-test/recovery/passed.json').read_text())
    protocol = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    if passed['protocol'] != protocol or not passed['continued_epoch_matches_exactly']:
        raise RuntimeError('Local recovery test does not match the protocol.')
    if json.loads((ROOT / 'configs/week4.json').read_text()) != reference['config']:
        raise RuntimeError('Settings differ from the tested configuration.')
    for name, expected in reference['sources'].items():
        with (ROOT / name).open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != expected:
                raise RuntimeError(f'Source differs from the recovery test: {name}')
    for name, version in reference['packages'].items():
        if importlib.metadata.version(name) != version:
            raise RuntimeError(f'Package version differs: {name}')
    data = ROOT / 'data/mvtec_ad'
    if (data / '.download-in-progress').exists():
        raise RuntimeError('Dataset download is incomplete.')
    inventory = json.loads((ROOT / 'docs/DATA_CHECK.json').read_text())
    for row in inventory['categories']:
        category = data / row['category']
        for split in ('train', 'test'):
            if len(list((category / split).glob('*/*.png'))) != row[split]:
                raise RuntimeError(f'Image count differs: {category.name}/{split}')
        for image in (category / 'test').glob('*/*.png'):
            if image.parent.name != 'good':
                mask = category / 'ground_truth' / image.parent.name / (image.stem + '_mask.png')
                if not mask.exists():
                    raise RuntimeError(f'Missing mask: {mask}')
    if shutil.disk_usage(ROOT).free < 8 * 1024**3:
        raise RuntimeError('Need at least 8 GiB free before starting another run.')
    print('Settings, source hashes, package versions, data counts, masks, and local recovery check passed.', flush=True)
    return reference, passed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    reference, passed = check()
    if args.check:
        return
    WORK.mkdir(parents=True, exist_ok=True)
    lock = (WORK / 'controller.lock').open('a+b')
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            raise RuntimeError('Another local Week 4 controller is already running.')
    STORE.mkdir(parents=True, exist_ok=True)
    manifest = STORE / 'study.json'
    if manifest.exists() and json.loads(manifest.read_text()) != reference:
        raise RuntimeError('Existing results use different settings.')
    write_json(manifest, reference)
    write_json(STORE / 'recovery/passed.json', passed)
    write_json(STORE / 'retention.json', {
        'temporary_features': str(WORK / 'cache'),
        'temporary_features_deleted_after_each_run': True,
        'optimizer_checkpoints_deleted_after_verified_run_completion': True,
        'final_results': 'Keep until verified backup and review are finished, then remove local bulk files.',
        'dataset_and_environment': 'Shared with the project. Remove after project work ends, not during this study.'})
    status = {'pid': os.getpid(), 'started_utc': datetime.now(timezone.utc).isoformat(),
              'state': 'running', 'results': str(STORE)}
    write_json(WORK / 'status.json', status)
    awake = False
    try:
        if os.name == 'nt':
            awake = bool(ctypes.windll.kernel32.SetThreadExecutionState(0x80000001))
            print(f'Prevent automatic sleep while training: {awake}', flush=True)
        for category in CATEGORIES:
            for seed in reference['config']['seeds']:
                if shutil.disk_usage(ROOT).free < 8 * 1024**3:
                    raise RuntimeError('Paused: less than 8 GiB free. Free space and rerun to resume.')
                status.update(category=category, seed=seed)
                write_json(WORK / 'status.json', status)
                print(f'Starting/checking {category}, seed {seed}', flush=True)
                subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/week4_colab.py'),
                                '--mode', 'run', '--store', str(STORE),
                                '--data', str(ROOT / 'data/mvtec_ad'),
                                '--scratch', str(WORK / 'cache'),
                                '--category', category, '--seed', str(seed)], cwd=ROOT, check=True)
        status.update(state='complete', finished_utc=datetime.now(timezone.utc).isoformat())
        print('All 45 runs finished. Final results are ready for review and backup.', flush=True)
    except BaseException as error:
        status.update(state='stopped', error=str(error))
        raise
    finally:
        write_json(WORK / 'status.json', status)
        if awake:
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
        lock.close()


if __name__ == '__main__':
    main()
