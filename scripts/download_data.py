"""Download MVTec AD without keeping a second copy of the archive."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
import urllib.request

URL = ('https://www.mydrive.ch/shares/150996/b52ecdcbf521176e9db9c731f2304b27/'
       'download/420938113-1629960298/mvtec_anomaly_detection.tar.xz')
SHA256 = 'cf4313b13603bec67abb49ca959488f7eedce2a9f7795ec54446c649ac98cd3d'


class CheckedStream:
    def __init__(self, response):
        self.response = response
        self.digest = hashlib.sha256()
        self.total = 0
        self.reported = 0

    def read(self, size=-1):
        chunk = self.response.read(size)
        self.digest.update(chunk)
        self.total += len(chunk)
        if self.total - self.reported >= 100 * 1024**2:
            print(f'Downloaded {self.total / 1024**3:.2f} GiB', flush=True)
            self.reported = self.total
        return chunk


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path('data/mvtec_ad'))
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    receipt = args.root / 'download.json'
    if receipt.exists():
        print(f'Download already completed: {receipt}')
        return
    if shutil.disk_usage(args.root).free < 7 * 1024**3:
        raise SystemExit('Need at least 7 GiB free. Choose another drive with --root.')
    incomplete = args.root / '.download-in-progress'
    incomplete.write_text('Download incomplete. Rerun scripts/download_data.py before training.')
    print('Downloading the original 15-category MVTec AD dataset.', flush=True)
    with urllib.request.urlopen(URL, timeout=120) as response:
        stream = CheckedStream(response)
        with tarfile.open(fileobj=stream, mode='r|xz') as archive:
            for member in archive:
                if shutil.disk_usage(args.root).free < 1024**3:
                    raise RuntimeError('Stopped: less than 1 GiB free. Download is incomplete.')
                archive.extract(member, args.root, filter='data')
        while stream.read(1024**2):
            pass
        if stream.digest.hexdigest() != SHA256:
            raise RuntimeError('Archive checksum mismatch. Do not use this download.')
    receipt.write_text(json.dumps({'url': URL, 'sha256': SHA256,
                                  'bytes': stream.total}, indent=2))
    incomplete.unlink()
    print(f'Checksum verified. Dataset saved to {args.root}', flush=True)


if __name__ == '__main__':
    main()
