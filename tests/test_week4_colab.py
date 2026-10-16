"""Check checkpoint recovery when a write is interrupted or damaged."""
import json
from pathlib import Path
import tempfile
import unittest

from scripts.week4_colab import save_checkpoint, load_checkpoint, torch


class CheckpointTests(unittest.TestCase):
    def test_corrupt_latest_uses_previous_and_keeps_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            for epoch in (25, 50, 75):
                save_checkpoint(folder, {'epoch': epoch, 'model': {'weight': torch.tensor([epoch])}})
            self.assertEqual(len(list(folder.glob('*.json'))), 2)
            newest = sorted(folder.glob('*.pt'))[-1]
            newest.write_bytes(b'interrupted copy')
            saved = load_checkpoint(folder)
            self.assertEqual(saved['epoch'], 50)
            self.assertEqual(saved['model']['weight'].item(), 50)

    def test_unfinished_first_checkpoint_does_not_restart_silently(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            (folder / 'epoch-0025.pt.partial').write_bytes(b'incomplete')
            with self.assertRaises(RuntimeError):
                load_checkpoint(folder)

    def test_uncommitted_copy_does_not_hide_last_checkpoint(self):
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_checkpoint(folder, {'epoch': 25, 'model': {'weight': torch.tensor([1.0])}})
            (folder / 'epoch-0050.pt.partial').write_bytes(b'incomplete')
            self.assertEqual(load_checkpoint(folder)['epoch'], 25)


if __name__ == '__main__':
    unittest.main()
