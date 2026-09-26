"""Check aggregation without mixing setup checks into baseline results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class SummaryTests(unittest.TestCase):
    def test_sample_std_and_smoke_exclusion(self):
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as folder:
            root = Path(folder)
            for seed, value in enumerate([0.5, 0.75, 1.0, 0.0]):
                run = root / 'resnet18/bottle' / f'seed_{seed}'
                run.mkdir(parents=True)
                (run / 'metrics.json').write_text(json.dumps({
                    'category': 'bottle', 'seed': seed, 'smoke': seed == 3,
                    'image_auroc': value, 'pixel_auroc': value,
                }))
                (run / 'config.json').write_text(json.dumps({
                    'backbone': 'resnet18', 'epochs_run': 500,
                    'batch_size': 8, 'size': 256, 'lr': 0.001,
                }))
            subprocess.run([sys.executable, str(ROOT / 'scripts/summarize.py'),
                            '--root', str(root)], check=True, capture_output=True)
            with (root / 'summary.csv').open() as handle:
                row = next(csv.DictReader(handle))
            self.assertEqual(int(row['runs']), 3)
            self.assertAlmostEqual(float(row['image_auroc_mean']), 0.75)
            self.assertAlmostEqual(float(row['image_auroc_std']), 0.25)


if __name__ == '__main__':
    unittest.main()
