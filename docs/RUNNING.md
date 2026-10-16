# Running and changing experiments

[Project overview](../README.md) | [Results](../reports/README.md)

## Read the work without running anything

The weekly READMEs, CSV tables and `HEATMAPS.md` pages are included in Git. They open directly on GitHub. The local HTML galleries and full model outputs are not included. A `file:///` link from another person's computer will not open their files on yours.

The source code is shared between weeks, so these are folders in one repository rather than separate repositories. This keeps one environment and avoids copies of the dataset. The recorded training files stay in their original locations because the saved studies check their source hashes.

## Set up a machine

Follow [Install](../README.md#install), then activate `.venv`. Commands below assume `python` points to that environment. The full training paths need an NVIDIA GPU. Week 5's study controller uses Windows process locking and sleep controls, so use Windows with CUDA for that controller as written. The older Colab notebooks are a separate route for Week 4, not a tested Week 5 deployment.

Download the original MVTec AD categories into:

```text
data/mvtec_ad/
  bottle/
    train/good/
    test/good/
    test/<defect>/
    ground_truth/<defect>/
  cable/
  ...
```

The shared category downloader skips existing category directories. An existing directory is not proof of a complete download. After any interrupted download, check its contents before training. The current downloader does not verify individual archive checksums. Our original local data used the combined archive checksum, which is a separate historical receipt.

Run the count and mask check:

```sh
python scripts/check_data.py
```

This checks the expected 3,629 training images, 1,725 test images and masks for every defective test image. It does not establish archive authenticity or check every pixel for corruption. Allow space for the data, environment, feature caches and results. The training controllers require at least 8 GiB free before preparing a category, in addition to files already present. The whole study needs more room as results accumulate.

## Week 3

Small installation check, with separate settings and outputs:

```sh
python run.py --smoke
```

One category with the actual Week 3 settings:

```sh
python week3.py --category bottle --seeds 0 1 2
```

Full study and results table:

```sh
python scripts/study.py
python scripts/week3_report.py
```

The full study uses [configs/week3.json](../configs/week3.json). It resumes saved runs and skips completed ones. Results go into `results/week3/`. Do not launch the category command and full controller together against the same output directory.

## Week 4 on a fresh machine

Use the shared runner directly, with a new store and scratch directory. The local convenience controller used for the original run expects a recovery-test directory from that machine, so it is not the fresh-clone entry point.

First create and check a two-epoch recovery test in separate Python processes:

```sh
python scripts/week4_colab.py --mode recovery-save --store results/week4-repeat --scratch work/week4-repeat
python scripts/week4_colab.py --mode recovery-check --store results/week4-repeat --scratch work/week4-repeat
```

Then run a single category and seed:

```sh
python scripts/week4_colab.py --mode run --store results/week4-repeat --scratch work/week4-repeat --category bottle --seed 0
```

Or start all 45 runs:

```sh
python scripts/week4_colab.py --mode study --store results/week4-repeat --scratch work/week4-repeat
```

The default dataset location is `data/mvtec_ad`. `--data PATH` selects another location. The settings come from [configs/week4.json](../configs/week4.json). Do not modify them partway through a study. The report scripts target the original `results/week4` folder, so adapt their input/output paths if summarizing a separate repeat.

## Week 5

The full controller requires the complete dataset inventory, even for its short smoke check. On a Windows CUDA machine, run the two-epoch controller test first:

```sh
python scripts/week5_study.py --smoke
```

This checks both arms and fresh-process recovery. Smoke outputs are separate from the study. The existing [faster-batch pilot](../reports/week5/fast_pilot.md) documents the original machine's memory and speed check. On a different GPU, measure a short run before estimating a full study. The smoke check proves operation, not useful final accuracy.

To start or resume the unchanged full comparison:

```sh
python scripts/week5_study.py
```

Settings are in [configs/week5_study.json](../configs/week5_study.json), and outputs are under `results/week5/study/`. The controller checks the source and settings against an existing study and refuses a mismatched resume. It retains model checkpoints at epochs 400, 450, 475 and 500.

To request a pause at an epoch boundary in PowerShell:

```powershell
New-Item -ItemType File -Path work/week5-study/PAUSE -Force
```

Wait for `work/week5-study/status.json` to say `paused`. To continue, remove that marker and run the controller again:

```powershell
Remove-Item -LiteralPath work/week5-study/PAUSE
python scripts/week5_study.py
```

## Trying a different idea

Keep the reported runs. Use a separate clone or experiment copy with a new output store for changed code or settings. The Week 5 runner currently fixes its output location, so a separate clone is the simplest route without changing the original runner. The dataset and environment can be prepared in that copy. Do not copy the old `results/week5/study` into it and try to resume with changed settings.

For a targeted Week 5 comparison, edit the configuration in that separate copy before the smoke check. Keep both arms, include bottle and seed 0 because the recovery check uses them, and keep retained epochs consistent with the final epoch. The full controller still requires the all-category dataset inventory. A reduced category list with the original 500 epochs is a smaller scope, not a full-dataset result.

The projection code explicitly implements 96 middle components and 32 components per fusion level. Changing `channels` alone does not implement a new compression budget. Changes to layer choice, channel allocation or normalization need matching changes in [week5_model.py](../scripts/week5_model.py), shape checks and another smoke test. The main result-analysis scripts expect the full original 15-category, three-seed, 500-epoch protocol and need adaptation for a changed experiment.

A useful proposal states the question, one intended change, matched control, categories, seeds, epoch count and main metric before running. A rerun to check a suspected issue is welcome, but report the original result too. Avoid repeatedly replacing weak seeds or choosing the best test checkpoint.

## Regenerate figures or inspect saved models

These commands need the original local dataset and saved outputs, not just a Git clone:

```sh
python scripts/week5_analysis.py
python scripts/compare_weeks.py
python scripts/week5_checkpoints.py
```

The first recalculates Week 5 metrics. The second makes cross-week tables and shareable galleries. Both use saved maps and CPU computation. The third replays a fixed four-category set of saved Week 5 checkpoints and requires CUDA. None starts training. Written interpretations are maintained separately from generated figures and tables.

For Week 3 and Week 4 checks, see their weekly reports. `week4_report.py` also replays selected models and therefore needs the local models and GPU.

## Where to look in the code

| File | Role |
| --- | --- |
| [run.py](../run.py) | Dataset loading, image and mask preprocessing, basic model check |
| [week3.py](../week3.py) | Baseline model construction, training and common evaluation |
| [study.py](../scripts/study.py) | Week 3 full-study controller |
| [week4_colab.py](../scripts/week4_colab.py) | Week 4 disk-cached training and checkpoint recovery |
| [week5_model.py](../scripts/week5_model.py) | PCA projection, feature alignment and compact flow |
| [week5_study.py](../scripts/week5_study.py) | Paired Week 5 study, recovery and checkpoint retention |
| [compare_weeks.py](../scripts/compare_weeks.py) | Same-image comparisons and cross-week statistics |

Run the code tests with `python -m unittest discover -s tests -v`. Passing tests and replay checks support implementation consistency. They do not establish paper-level accuracy.

## Keeping the project after training

Temporary feature caches are removed by the study controllers. Final weights, maps, scores and reports are retained for review. Do not delete those local outputs until the team has a verified backup and no longer needs the replays. Git contains the readable reports and selected figures, not a backup of every trained model.
