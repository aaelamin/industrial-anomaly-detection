"""Check category averaging and completion rules with small synthetic results."""
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = 'bottle cable capsule carpet grid hazelnut leather metal_nut pill screw tile toothbrush transistor wood zipper'.split()


class ReportTests(unittest.TestCase):
    def test_macro_std_is_across_seed_means(self):
        (ROOT / 'work').mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT / 'work') as folder:
            base = Path(folder)
            for index, category in enumerate(CATEGORIES):
                for seed in [0, 1, 2]:
                    variant = seed if index % 2 == 0 else 2 - seed
                    abnormal = [[0, 1], [0.5, 1.5], [2, 3]][variant]
                    auc = [0.5, 0.75, 1][variant]
                    run = base / 'runs' / category / f'seed_{seed}'
                    run.mkdir(parents=True)
                    result = {'epochs': 500, 'category': category, 'seed': seed,
                              'protocol_hash': 'synthetic-test-only', 'test_images': 4,
                              'image_auroc': auc, 'pixel_auroc': auc,
                              'pixel_average_precision': auc, 'training_seconds': 1,
                              'defects': {'broken': {'images': 2, 'image_auroc': auc}}}
                    (run / 'metrics.json').write_text(json.dumps(result))
                    (run / 'config.json').write_text(json.dumps({'protocol_hash': 'synthetic-test-only'}))
                    with (run / 'scores.csv').open('w', newline='') as handle:
                        writer = csv.writer(handle)
                        writer.writerow(['image', 'label', 'score', 'defect', 'mask_fraction'])
                        for i, value in enumerate([0, 1] + abnormal):
                            label = int(i >= 2)
                            defect = 'broken' if label else 'good'
                            writer.writerow([f'test/{defect}/{i}.png', label, value, defect, label * 0.1])
            command = [sys.executable, str(ROOT / 'scripts/week3_report.py'),
                       '--root', str(base / 'runs'), '--output', str(base / 'report')]
            subprocess.run(command, check=True, capture_output=True)
            status = json.loads((base / 'report/status.json').read_text())
            self.assertTrue(status['complete'])
            self.assertAlmostEqual(status['image_auroc']['mean'], 0.75)
            self.assertAlmostEqual(status['image_auroc']['std'], 1 / 60)
            (base / 'runs/zipper/seed_2/metrics.json').unlink()
            subprocess.run(command, check=True, capture_output=True)
            status = json.loads((base / 'report/status.json').read_text())
            self.assertFalse(status['complete'])
            self.assertEqual(status['completed_runs'], 44)
            self.assertNotIn('image_auroc', status)


if __name__ == '__main__':
    unittest.main()
