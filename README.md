# Industrial Anomaly Detection

CISC 473, Project 16

Industrial anomaly detection is about finding defects without needing examples of every possible defect during training. The model learns from normal products, then looks for differences in new images. It can give a score for the whole image and produce a heatmap showing where a defect might be.

For this project, we are starting with FastFlow on MVTec AD. FastFlow takes image features from a pretrained network and learns their distribution using normalizing flows. We will compare feature extractors and combinations of feature layers, then test how the method performs on reflective battery surfaces using BatterySurAD. The later experiments will help us decide what improvement to investigate.

## Schedule

The schedule from our proposal:

| Week | Tasks | Planned lead |
| --- | --- | --- |
| 1 | Finish reading the four core papers | Jagger |
| 2 | Download MVTec; baseline repo setup; proposal due | Tomer |
| 3 | Run the baseline on all 15 categories; compute AUROC | Adam |
| 4 | Backbone ablation: ResNet-18 vs. WideResNet-50 | Jagger |
| 5 | Multi-scale flow implementation | Tomer |
| 6 | Multi-scale results and anomaly-map visualizations; midterm due | Adam |
| 7 | Domain-transfer experiment | Jagger |
| 8 | Failure analysis and per-category AUROC breakdown | Tomer |
| 9 | Report and anomaly-localization heatmaps | Adam |
| 10 | Final submission | Jagger |

EfficientNet-B4 is also included in the broader backbone comparison from the evaluation plan.

## Install

From the repository folder:

```sh
python -m pip install uv
python scripts/setup.py --device cuda
```

This creates a Python 3.12 environment in `.venv` and installs the packages. The GPU setup uses PyTorch 2.7.1 with CUDA 12.6 and needs an NVIDIA GPU with a compatible driver. Use `--device cpu` if you do not have an NVIDIA GPU, including on a Mac.

Activate the environment in PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use `.venv\Scripts\python.exe` instead of `python` in the commands below. On Linux or macOS, activate with `source .venv/bin/activate`. So far, the setup has been tested on Windows with GPU and CPU execution.

## Download the data

```sh
python scripts/download_data.py
```

The script downloads MVTec AD to `data/mvtec_ad` and checks the archive checksum. It needs at least 7 GiB free before starting, with additional space needed for training results.

To store the dataset elsewhere, pass `--root PATH` to the download script and `--data PATH` when running `week3.py`.

MVTec AD uses the [CC BY-NC-SA 4.0 license](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The dataset is not included in this repository. Example heatmaps in the reports use images from MVTec AD.

## Run FastFlow

To run all 15 categories with three seeds each:

```sh
python scripts/study.py
```

The script runs one experiment at a time and updates the report after each run. If it stops, run the same command again. Finished runs are skipped, and unfinished runs resume from a checkpoint saved every 25 epochs. Full training takes several hours on a GPU, so keep the laptop plugged in and the lid open.

To run one category:

```sh
python week3.py --category bottle --seeds 0 1 2
python scripts/week3_report.py
```

Run either the full study or individual categories, not both at once using the same output folders. The settings are in [configs/week3.json](configs/week3.json), with more detail in the [experiment protocol](docs/WEEK3_PROTOCOL.md). Use a separate output directory for experiments with changed settings.

`run.py --smoke` is a short installation check with different settings. Its results are not included in the baseline tables.

## Results and saved files

The results tables are in `reports/week3/`. Each training run saves its weights, checkpoint, loss history, image scores, anomaly maps and heatmaps under `results/week3/<category>/seed_<n>/`.

The dataset, environment and full training outputs are excluded from Git. The reports and selected example heatmaps are included. For a running study, check `results/week3/study_status.json` and the run's `progress.json`.

## Tests

```sh
python -m unittest discover -s tests -v
```

These check the training calculations, checkpoint resume and result summaries. The [validation report](reports/week3/validation.md) also explains how to recheck the saved results. That requires the dataset and training outputs, plus a CUDA GPU to replay the selected models.

## Week 3 progress

We ran FastFlow with ResNet-18 on all 15 categories using seeds 0, 1 and 2. Each run trained for 500 epochs on normal images only. We used the final checkpoint for evaluation rather than selecting the epoch with the best test score.

We used Anomalib 2.6.2 for the FastFlow model and loss so we could build on an existing implementation instead of writing the architecture from scratch. Our scripts handle the training runs, checkpoints and evaluation. Anomalib's version is based on the unofficial gathierry FastFlow implementation.

| Metric | Our result, mean ± std | FastFlow paper, ResNet-18 |
| --- | ---: | ---: |
| Image AUROC | 90.17 ± 0.99% | 97.9% |
| Pixel AUROC | 95.41 ± 0.59% | 97.2% |

For each seed, we average AUROC across the 15 categories. The table reports the mean and standard deviation across those three averages.

Our overall scores are lower than the paper's, particularly for image detection on hazelnut and screw. We checked the saved scores and masks for all 45 runs and replayed three models. Those checks passed, along with the four automated tests. Some difficult normal images have strong responses in the background, which can affect the maximum-pixel score used for image detection.

The unofficial implementation reports about 95.6% pixel AUROC at the final epoch, close to our 95.41%. Its best-epoch results are higher. This gives some context for the pixel result, but we have not fully explained the gap from the paper, especially at image level. We will keep these results as our baseline for later comparisons and report that limitation.

- [Full results by category](reports/week3/week3.md)
- [Checks, failure analysis and heatmaps](reports/week3/validation.md)
- [Per-category CSV](reports/week3/per_category.csv)
- [Per-defect CSV](reports/week3/per_defect.csv)

## References

- [FastFlow paper](https://arxiv.org/abs/2111.07677)
- [Anomalib](https://github.com/open-edge-platform/anomalib)
- [Unofficial FastFlow implementation and results](https://github.com/gathierry/FastFlow)
