# Industrial Anomaly Detection

CISC 473, Project 16

Industrial anomaly detection is about finding defects without needing examples of every possible defect during training. We train on normal products, then ask whether a new image is unusual and where the defect might be.

Our starting method is FastFlow on the original 15-category MVTec AD dataset. We compared two backbone configurations, then tested a compact model that combines feature levels before scoring them. Later work will consider domain transfer to reflective battery surfaces. That experiment has not been run yet.

## Start here

You do not need to install anything to read the results or view the heatmaps on GitHub.

| Read | What is in it |
| --- | --- |
| [Week 3](reports/week3/README.md) | What FastFlow does, how we set up the baseline, results and conclusions |
| [Week 4](reports/week4/README.md) | Why we tried WideResNet, what improved and what remained weak |
| [Week 5](reports/week5/README.md) | Why we tried feature fusion, how we controlled the comparison and what happened |
| [Comparison and midterm discussion](reports/comparison/README.md) | All experiments together, matched heatmaps and what we can reasonably conclude |

Each weekly folder includes the explanation, category results and a `HEATMAPS.md` gallery. The [cross-week gallery](reports/comparison/HEATMAPS.md) shows the same images for every configuration and all three seeds.

Week 4 is our strongest configuration so far. The Week 5 fusion experiment finished all 90 runs but did not improve overall performance. The reports include the losses as well as the gains. The gap from the published image-level results remains unresolved.

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

This creates the pinned Python 3.12 environment in `.venv`. CUDA training needs a compatible NVIDIA GPU. Use `--device cpu` for the basic CPU environment. The full Week 5 controller currently targets Windows with CUDA, so the environment installing elsewhere does not mean that controller is portable.

In PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

If activation is blocked, use `.venv\Scripts\python.exe` in place of `python`. On Linux or macOS, use `source .venv/bin/activate` for the environment. The [run guide](docs/RUNNING.md) explains the supported routes and commands.

## Download the data

```sh
python scripts/download_data.py
```

The downloader uses separate category archives after the original combined download link stopped working. This is the original MVTec AD dataset, not MVTec AD 2. Keep the folder layout `data/mvtec_ad/<category>/train`, `test` and `ground_truth`. The current category downloader does not verify archive checksums. Check the image counts and masks before a new study, as described in the [run guide](docs/RUNNING.md).

If a link is unavailable, use the [official MVTec AD download page](https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads) and extract each category into that layout. The dataset uses [CC BY-NC-SA 4.0](https://www.mvtec.com/research-teaching/datasets/mvtec-ad). The same attribution applies to the example images in our reports.

## Run or change an experiment

Start with [the run guide](docs/RUNNING.md). It separates small checks from full studies, explains where files go and shows which settings to change. A full command starts substantial work, so use the short checks first on a new machine.

For the original Week 3 study:

```sh
python scripts/study.py
```

This runs all categories and seeds, skips completed outputs and resumes unfinished runs from saved checkpoints. The Week 4 and Week 5 commands and their recovery requirements are in the guide. Changed experiments need their own output locations so the published runs remain available for comparison.

## Repository layout

| Folder | Purpose |
| --- | --- |
| `reports/` | Readable weekly explanations, results, selected heatmaps and comparisons |
| `configs/` | Recorded experiment settings |
| `scripts/` | Setup, training, evaluation and report generation |
| `docs/` | Detailed protocols and instructions for running or changing experiments |
| `tests/` | Tests for training calculations, recovery and reporting |
| `notebooks/` | Earlier Colab setup and benchmark notebooks |
| `data/`, `results/`, `work/` | Local dataset, full outputs and temporary working files, excluded from Git |

A clone includes the tables and published heatmaps. It does not include the dataset, environment, trained models or full prediction maps. The [results index](reports/README.md) explains where to find each result.

## References

- [FastFlow](https://arxiv.org/abs/2111.07677), our baseline method
- [Anomalib](https://github.com/open-edge-platform/anomalib), the implementation used here
- [CFLOW-AD](https://arxiv.org/abs/2107.12571), related conditional multi-scale flow work
- [MSFlow](https://arxiv.org/abs/2308.15300), a different published fusion architecture and a possible comparison

We have not run CFLOW-AD or MSFlow. Our Week 5 model is a project experiment, not a reproduction of either method.
