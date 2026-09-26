"""Create the environment using uv, then install the locked packages."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    uv = shutil.which('uv')
    if uv is None:
        raise SystemExit('Install uv first: python -m pip install uv')
    os.environ['UV_PYTHON_INSTALL_DIR'] = str(root / '.python')
    os.environ['UV_CACHE_DIR'] = str(root / '.uv-cache')
    python = root / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
    if not python.exists():
        subprocess.run([uv, 'venv', '--python', '3.12', '.venv'], check=True)
    subprocess.run([str(python), '-c',
                    'import sys; assert sys.version_info[:2] == (3, 12), '
                    '"Existing .venv must use Python 3.12"'], check=True)
    backend = 'cu126' if args.device == 'cuda' else 'cpu'
    if args.device == 'cuda' and sys.platform == 'darwin':
        raise SystemExit('Use --device cpu on macOS.')
    suffix = '' if sys.platform == 'darwin' else '+' + backend
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '--no-cache',
                    'torch==2.7.1' + suffix, 'torchvision==0.22.1' + suffix, '--index',
                    f'https://download.pytorch.org/whl/{backend}',
                    '--index-strategy', 'unsafe-best-match'], check=True)
    # Keep the chosen CPU/CUDA build instead of resolving PyTorch a second time.
    subprocess.run([uv, 'pip', 'install', '--python', str(python), '--no-cache',
                    '--no-deps', '-r', 'requirements.txt'], check=True)
    subprocess.run([uv, 'pip', 'check', '--python', str(python), '--no-cache'], check=True)
    subprocess.run([str(python), '-c',
                    'import torch; print("PyTorch:", torch.__version__); '
                    'print("CUDA available:", torch.cuda.is_available())'], check=True)
    if args.device == 'cuda':
        subprocess.run([str(python), '-c',
                        'import torch; assert torch.cuda.is_available(), '
                        '"CUDA unavailable: check NVIDIA driver or use --device cpu"'], check=True)


if __name__ == '__main__':
    main()
