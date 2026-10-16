"""Queue small batches and transfer finished results before reusing Drive space."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()
BUDGET = 5_000_000_000
RESERVE = 2_500_000_000


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def run_path(store, key):
    if not re.fullmatch(r'(' + '|'.join(CATEGORIES) + r')/seed_[012]', key):
        raise ValueError('Invalid run name')
    path = (store / 'runs' / key).resolve()
    if not path.is_relative_to((store / 'runs').resolve()):
        raise ValueError('Run path escapes results folder')
    return path


def checked_complete(folder, protocol):
    complete = json.loads((folder / 'complete.json').read_text())
    if complete['protocol'] != protocol:
        raise ValueError('Result protocol differs')
    for name, expected in complete['files'].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder.resolve()) or sha(path) != expected:
            raise ValueError(f'Result missing or changed: {name}')
    return complete


def archived(store):
    path = store / 'archived.json'
    return json.loads(path.read_text()) if path.exists() else {}


def pending(store):
    ledger = archived(store)
    return [p.parent.relative_to(store / 'runs').as_posix()
            for p in sorted((store / 'runs').glob('*/seed_*/complete.json'))
            if p.parent.relative_to(store / 'runs').as_posix() not in ledger]


def queue(store):
    protocol = digest(json.loads((store / 'study.json').read_text()))
    passed = json.loads((store / 'recovery/passed.json').read_text())
    if passed['protocol'] != protocol or not passed['continued_epoch_matches_exactly']:
        raise ValueError('Recovery check has not passed for these settings')
    if not (store / 'recovery/fresh_runtime_passed.json').exists():
        raise ValueError('Complete the fresh-runtime check first')
    if (ROOT / 'data/mvtec_ad/.download-in-progress').exists():
        raise ValueError('Dataset download is incomplete')
    ledger = archived(store)
    for category in CATEGORIES:
        for seed in range(3):
            key = f'{category}/seed_{seed}'
            if key in ledger:
                if ledger[key]['protocol'] != protocol:
                    raise ValueError('Archived run protocol differs')
                continue
            folder = run_path(store, key)
            if (folder / 'complete.json').exists():
                checked_complete(folder, protocol)
                continue
            if len(pending(store)) >= 3:
                print('Batch ready: download and verify the three finished runs before continuing.', flush=True)
                return
            used = sum(p.stat().st_size for p in store.rglob('*') if p.is_file())
            if used + RESERVE > BUDGET:
                print(f'Paused for storage: study folder uses {used / 1e9:.2f} GB. '
                      'Transfer finished runs before continuing. Nothing was deleted.', flush=True)
                return
            subprocess.run([sys.executable, '-u', str(ROOT / 'scripts/week4_colab.py'),
                            '--mode', 'run', '--store', str(store),
                            '--category', category, '--seed', str(seed)], check=True, cwd=ROOT)
    print('All 45 runs finished. Export any remaining results.', flush=True)


def export(store, archive):
    keys = pending(store)
    if not keys:
        raise ValueError('No finished runs are waiting for transfer')
    protocol = digest(json.loads((store / 'study.json').read_text()))
    archive.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED, compresslevel=1) as z:
        z.write(store / 'study.json', 'study.json')
        for key in keys:
            folder = run_path(store, key)
            complete = checked_complete(folder, protocol)
            for name in list(complete['files']) + ['complete.json']:
                z.write(folder / name, f'runs/{key}/{name}')
    # Read the full archive back before offering it for download.
    verify(archive, archive.with_suffix('.verification.json'))
    print(f'Batch archive ready: {archive}. No Drive results were deleted.', flush=True)


def verify(archive, receipt):
    if archive.resolve() == receipt.resolve():
        raise ValueError('Receipt must not overwrite the archive')
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate archive members')
        study = json.loads(z.read('study.json'))
        protocol = digest(study)
        runs = {}
        expected_names = {'study.json'}
        for name in names:
            if not name.endswith('/complete.json'):
                continue
            parts = name.split('/')
            if len(parts) != 4 or parts[0] != 'runs':
                raise ValueError('Invalid archive layout')
            key = '/'.join(parts[1:3])
            run_path(Path('.'), key)
            complete = json.loads(z.read(name))
            if complete['protocol'] != protocol:
                raise ValueError('Protocol mismatch in archive')
            expected_names.add(name)
            for relative, expected in complete['files'].items():
                member = f'runs/{key}/{relative}'
                if '..' in Path(relative).parts or Path(relative).is_absolute():
                    raise ValueError('Invalid result filename')
                with z.open(member) as handle:
                    actual = hashlib.file_digest(handle, 'sha256').hexdigest()
                if actual != expected:
                    raise ValueError(f'Archive checksum mismatch: {member}')
                expected_names.add(member)
            runs[key] = {'protocol': protocol, 'complete_sha256': digest(complete)}
        if not runs or set(names) != expected_names:
            raise ValueError('Missing runs or unexpected archive contents')
    write_json(receipt, {'archive_sha256': sha(archive), 'archive_name': archive.name, 'runs': runs})
    print(f'Verified {len(runs)} runs. Keep {archive}; upload {receipt.name} in step 10.', flush=True)


def release(store, receipt):
    proof = json.loads(receipt.read_text())
    if not re.fullmatch('[0-9a-f]{64}', proof['archive_sha256']):
        raise ValueError('Invalid verification receipt')
    protocol = digest(json.loads((store / 'study.json').read_text()))
    ledger = archived(store)
    for key, record in proof['runs'].items():
        folder = run_path(store, key)
        if record['protocol'] != protocol:
            raise ValueError('Transfer receipt protocol differs')
        if key in ledger:
            if ledger[key]['archive_sha256'] != proof['archive_sha256']:
                raise ValueError('Run was already transferred in another archive')
            continue
        complete = checked_complete(folder, protocol)
        if digest(complete) != record['complete_sha256']:
            raise ValueError('Drive result differs from the downloaded copy')
    for key, record in proof['runs'].items():
        ledger[key] = {**record, 'archive_sha256': proof['archive_sha256'],
                       'archive_name': proof['archive_name']}
    write_json(store / 'archived.json', ledger)
    for key in proof['runs']:
        folder = run_path(store, key)
        if folder.exists():
            shutil.rmtree(folder)
    print('Transferred runs recorded; their Drive copies removed. You can resume step 7.', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['queue', 'export', 'verify', 'release'])
    parser.add_argument('--store', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.mode == 'verify':
        if not args.archive or not args.receipt:
            parser.error('verify requires --archive and --receipt')
        verify(args.archive, args.receipt)
    else:
        if not args.store:
            parser.error('--store is required')
        if args.mode == 'queue':
            queue(args.store)
        elif args.mode == 'export':
            if not args.archive: parser.error('--archive is required')
            export(args.store, args.archive)
        else:
            if not args.receipt: parser.error('--receipt is required')
            release(args.store, args.receipt)


if __name__ == '__main__':
    main()
