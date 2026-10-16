"""Check category counts and mask presence before starting a study."""
import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=ROOT/'data/mvtec_ad')
    args = parser.parse_args()
    expected = json.loads((ROOT/'docs/DATA_CHECK.json').read_text())['categories']
    errors = []
    for row in expected:
        category = args.root/row['category']
        for split in ('train','test'):
            images = list((category/split).glob('*/*.png'))
            if len(images)!=row[split]:
                errors.append(f"{row['category']}/{split}: expected {row[split]}, found {len(images)}")
            for image in images:
                if split=='train' and image.parent.name!='good':
                    errors.append(f'Unexpected non-normal training image: {image}')
                if split=='test' and image.parent.name!='good':
                    mask=category/'ground_truth'/image.parent.name/(image.stem+'_mask.png')
                    if not mask.is_file():
                        errors.append(f'Missing mask: {mask}')
    if (args.root/'.download-in-progress').exists():
        errors.append('Download-in-progress marker exists. Resolve the incomplete download before training.')
    if errors:
        raise SystemExit('\n'.join(errors))
    print('All 15 categories have the expected image counts and required masks.')
    print('This checks file presence, not archive checksums or image contents.')


if __name__=='__main__':
    main()
