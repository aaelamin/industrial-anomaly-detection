"""Download MVTec AD category archives and remove each after extraction."""
import argparse
from pathlib import Path
import shutil
import tarfile
import urllib.request

CATEGORY_URLS = {
    "bottle": "https://www.mydrive.ch/shares/150452/132a93367fb17cdf968dfb5c4013f6e7/download/420937370-1629958698/bottle.tar.xz",
    "cable": "https://www.mydrive.ch/shares/150453/10f960e07fec2838b2cd512586633a32/download/420937413-1629958794/cable.tar.xz",
    "capsule": "https://www.mydrive.ch/shares/150454/e0ce6dd74eb150f46c0d98131b3703f2/download/420937454-1629958872/capsule.tar.xz",
    "carpet": "https://www.mydrive.ch/shares/150455/eac7fbce84d93a5094e13f391170eca4/download/420937484-1629959013/carpet.tar.xz",
    "grid": "https://www.mydrive.ch/shares/150456/bb0b2e3dc804ccb8b4485e01f8e4493b/download/420937487-1629959044/grid.tar.xz",
    "hazelnut": "https://www.mydrive.ch/shares/150457/51d49f65e84bdc100ff8035d7dd783ea/download/420937545-1629959162/hazelnut.tar.xz",
    "leather": "https://www.mydrive.ch/shares/150458/923030ce14e10a7d147e95d0f8885f6b/download/420937607-1629959262/leather.tar.xz",
    "metal_nut": "https://www.mydrive.ch/shares/150459/c68856a21dca589b0f8ff6d4ee0f18f4/download/420937637-1629959294/metal_nut.tar.xz",
    "pill": "https://www.mydrive.ch/shares/150460/d4f1c04da67034ccc8d8b5fa3e73f244/download/420938129-1629960351/pill.tar.xz",
    "screw": "https://www.mydrive.ch/shares/150461/242f454cc6385e5693c4fd4b94567d1e/download/420938130-1629960389/screw.tar.xz",
    "tile": "https://www.mydrive.ch/shares/150462/5479f0fdc97bc6fa16eab0cb0cf0109f/download/420938133-1629960456/tile.tar.xz",
    "toothbrush": "https://www.mydrive.ch/shares/150463/895a1a5fb84b958f07417784b060434b/download/420938134-1629960477/toothbrush.tar.xz",
    "transistor": "https://www.mydrive.ch/shares/150464/99b8ea0332438d64fcc2475ee73f9c29/download/420938166-1629960554/transistor.tar.xz",
    "wood": "https://www.mydrive.ch/shares/150465/d5b4115b720cdb54d217e75636e6e374/download/420938383-1629960649/wood.tar.xz",
    "zipper": "https://www.mydrive.ch/shares/150466/bd155b557520edaf692d9bfdb915c24a/download/420938385-1629960680/zipper.tar.xz",
}


def download_category(name: str, url: str, root: Path):
    print(f"Downloading {name}...")
    archive_path = root / f"{name}.tar.xz"
    urllib.request.urlretrieve(url, archive_path)
    print(f"Extracting {name}...")
    with tarfile.open(archive_path, "r:xz") as archive:
        archive.extractall(root, filter='data')
    archive_path.unlink()
    print(f"Finished {name}")


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
    for name, url in CATEGORY_URLS.items():
        category_dir = args.root / name
        if category_dir.exists():
            print(f"Skipping {name} (already exists)")
            continue
        download_category(name, url, args.root)
    incomplete.unlink()
    print(f'Dataset saved to {args.root}. Run scripts/check_data.py before training.', flush=True)


if __name__ == '__main__':
    main()
