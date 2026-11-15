# Environment setup details

The observed interpreter is **Python 3.12.13 on Windows**. The complete shared dependency pins are in `legacy/requirements.txt`; the shorter `additional/requirements-observed.txt` lists the model-stack identities checked by the additional-study release. Use the complete shared file for a new environment, not only the shorter list. No clean installation has yet been executed for this combined package.

Example commands from the package directory, after creating and activating a Python 3.12 environment:

```powershell
python -m pip install torch==2.7.1+cu126 torchvision==0.22.1+cu126 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r legacy/requirements.txt
python -m pip install psutil==7.0.0
python -B reproduce.py preflight --config host-config.json
```

These are the recorded environment's pins and a proposed installation route, not evidence that all wheels are presently available or that another GPU produces bitwise identical results. Do not silently substitute package versions or weights if installation fails. Preserve the failure, document any new environment, and qualify it separately before claiming comparability.

The runner does not download datasets or weights. See `host-config.example.json` for the exact required weight filenames and SHA-256 identities. The two legacy encoders also require those files under the configured `torch_home/hub/checkpoints/`. A system without the required disk/RAM/GPU headroom should not start the full study merely because preflight can verify metadata.
