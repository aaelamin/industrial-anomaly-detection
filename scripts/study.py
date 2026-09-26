"""Run the 45 fixed experiments in order and update the report after each one."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()
SOURCES = ['week3.py', 'run.py', 'configs/week3.json', 'requirements.txt', 'scripts/week3_report.py']


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    os.chdir(ROOT)
    out = ROOT / 'results/week3'
    out.mkdir(parents=True, exist_ok=True)
    # The OS releases this lock if the process exits or the machine restarts.
    lock = (out / 'study.lock').open('a+b')
    if lock.tell() == 0:
        lock.write(b'0')
        lock.flush()
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        import fcntl
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    hashes = {name: digest(ROOT / name) for name in SOURCES}
    snapshot = out / 'source'
    snapshot.mkdir(exist_ok=True)
    manifest = snapshot / 'hashes.json'
    if manifest.exists() and json.loads(manifest.read_text()) != hashes:
        raise RuntimeError('The study source has changed. Review before resuming.')
    for name in SOURCES:
        target = snapshot / name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(ROOT / name, target)
    manifest.write_text(json.dumps(hashes, indent=2))
    status_path = out / 'study_status.json'
    status = {'pid': os.getpid(), 'state': 'running', 'started_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}
    try:
        for category in CATEGORIES:
            for seed in [0, 1, 2]:
                if any(digest(ROOT / name) != value for name, value in hashes.items()):
                    raise RuntimeError('Study source changed during execution.')
                status.update({'category': category, 'seed': seed})
                status_path.write_text(json.dumps(status, indent=2))
                command = [sys.executable, '-u', 'week3.py', '--category', category, '--seeds', str(seed)]
                print(' '.join(command), flush=True)
                subprocess.run(command, check=True)
                subprocess.run([sys.executable, 'scripts/week3_report.py'], check=True)
        status['state'] = 'complete'
    except BaseException as error:
        status.update({'state': 'failed', 'error': str(error)})
        raise
    finally:
        status['updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        status_path.write_text(json.dumps(status, indent=2))
        lock.close()


if __name__ == '__main__':
    main()
