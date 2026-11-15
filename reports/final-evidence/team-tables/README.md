# Complete result tables for the report team

315 MVTec runs: seven configurations, all 15 categories and three seeds. Tables use saved fixed-endpoint metrics. These are development/benchmark comparisons, not results on the reserved AeBAD images.

| Configuration | Image AUROC (%) | Pixel AUROC (%) | Pixel AP (%) |
|---|---:|---:|---:|
| Original ResNet-18 | 90.17 ± 0.99 | 95.41 ± 0.59 | 39.69 ± 1.04 |
| Matched ResNet-18 | 89.72 ± 0.65 | 94.90 ± 0.21 | 39.11 ± 0.54 |
| WideResNet-50-2 | 93.17 ± 2.14 | 97.19 ± 0.49 | 52.99 ± 2.95 |
| EfficientNet-B4 | 72.80 ± 3.13 | 75.75 ± 2.93 | 15.13 ± 1.46 |
| Compact middle layer | 71.76 ± 3.39 | 73.89 ± 1.41 | 14.57 ± 1.68 |
| Compact multi-scale fusion | 67.55 ± 1.66 | 69.80 ± 1.68 | 13.79 ± 1.04 |
| PatchCore reference | 98.40 ± 0.05 | 97.88 ± 0.02 | 56.13 ± 0.06 |

Each cell is the mean ± sample SD of three seed-level macro scores. Within a seed, each category has equal weight. SD describes repeatability on the same images; it is not a confidence interval over independent industrial objects.

## Planned descriptive contrasts

| Difference | Pixel AP change (points) | Seed changes (points) |
|---|---:|---|
| fusion minus middle | -0.78 ± 2.23 | -3.20, +1.19, -0.32 |
| resnet minus week3 | -0.59 ± 1.46 | +0.65, -2.20, -0.21 |
| week4 minus resnet | +13.88 ± 3.49 | +9.92, +16.50, +15.21 |
| patchcore minus week3 | +16.43 ± 1.01 | +17.00, +15.26, +17.04 |
| patchcore minus resnet | +17.02 ± 0.59 | +16.35, +17.46, +17.25 |
| patchcore minus week4 | +3.14 ± 2.90 | +6.43, +0.96, +2.04 |
| patchcore minus efficientnet | +40.99 ± 1.52 | +39.30, +41.42, +42.26 |
| patchcore minus middle | +41.56 ± 1.74 | +39.57, +42.36, +42.75 |
| patchcore minus fusion | +42.34 ± 1.02 | +42.77, +41.17, +43.07 |

Paired seed differences are descriptive; equal seed numbers across architectures do not imply identical optimization randomness. The compact fusion/control pair explicitly shares initialization and training order. There are no significance claims or additional model-selection decisions here.

The original ResNet and matched ResNet share weight identity, batch size and microbatch; their coupling-kernel recipes differ. They also come from different recorded execution routes, so this comparison alone does not isolate every implementation detail. WideResNet and EfficientNet use different pretrained weights. PatchCore is a separate published-method reference.

## Per-category pixel AP

| Category | Original ResNet-18 | Matched ResNet-18 | WideResNet-50-2 | EfficientNet-B4 | Compact middle layer | Compact multi-scale fusion | PatchCore reference |
|---|---:|---:|---:|---:|---:|---:|---:|
| bottle | 67.64 ± 1.76 | 67.49 ± 4.30 | 77.05 ± 1.74 | 15.05 ± 3.35 | 36.39 ± 1.28 | 42.31 ± 2.79 | 76.04 ± 0.03 |
| cable | 31.00 ± 3.99 | 19.17 ± 3.24 | 60.95 ± 5.88 | 3.12 ± 0.45 | 3.84 ± 0.79 | 2.20 ± 0.52 | 65.55 ± 0.24 |
| capsule | 26.97 ± 1.13 | 27.77 ± 2.03 | 35.78 ± 1.54 | 23.92 ± 0.59 | 19.67 ± 0.76 | 23.71 ± 2.54 | 46.60 ± 0.08 |
| carpet | 63.05 ± 1.47 | 64.65 ± 1.31 | 59.43 ± 4.59 | 58.23 ± 6.48 | 16.82 ± 9.83 | 8.87 ± 8.90 | 62.28 ± 0.33 |
| grid | 37.22 ± 3.24 | 35.10 ± 5.58 | 43.83 ± 1.65 | 7.88 ± 6.23 | 3.07 ± 1.34 | 1.33 ± 0.70 | 32.73 ± 0.50 |
| hazelnut | 23.82 ± 1.15 | 24.68 ± 6.55 | 40.06 ± 3.26 | 8.90 ± 4.46 | 5.64 ± 1.75 | 2.10 ± 0.37 | 55.72 ± 0.35 |
| leather | 45.33 ± 4.69 | 45.41 ± 4.38 | 64.00 ± 5.32 | 17.43 ± 4.39 | 11.41 ± 10.24 | 4.03 ± 1.56 | 44.17 ± 0.13 |
| metal_nut | 55.23 ± 9.23 | 59.54 ± 3.65 | 81.80 ± 0.66 | 10.90 ± 1.16 | 13.27 ± 1.63 | 15.68 ± 1.56 | 88.16 ± 0.05 |
| pill | 38.28 ± 3.24 | 36.04 ± 4.61 | 42.73 ± 20.95 | 5.70 ± 1.97 | 7.10 ± 3.47 | 14.49 ± 2.63 | 77.45 ± 0.18 |
| screw | 5.56 ± 1.16 | 4.23 ± 1.33 | 14.22 ± 2.98 | 1.25 ± 0.29 | 0.89 ± 0.21 | 0.77 ± 0.08 | 32.08 ± 0.56 |
| tile | 44.19 ± 1.91 | 40.14 ± 3.85 | 68.43 ± 2.87 | 10.33 ± 0.90 | 14.12 ± 1.24 | 9.01 ± 0.86 | 51.14 ± 0.17 |
| toothbrush | 24.34 ± 1.30 | 21.05 ± 0.48 | 34.48 ± 0.58 | 23.34 ± 0.36 | 29.65 ± 4.84 | 34.79 ± 6.24 | 55.51 ± 0.22 |
| transistor | 58.87 ± 6.68 | 60.08 ± 2.07 | 68.51 ± 4.73 | 11.50 ± 4.08 | 6.34 ± 1.74 | 13.65 ± 7.51 | 63.57 ± 0.19 |
| wood | 39.25 ± 2.33 | 43.80 ± 2.51 | 54.95 ± 5.70 | 18.44 ± 18.08 | 18.42 ± 2.48 | 5.90 ± 2.68 | 44.45 ± 0.44 |
| zipper | 34.66 ± 4.54 | 37.45 ± 1.13 | 48.56 ± 0.56 | 10.98 ± 2.30 | 31.85 ± 7.86 | 27.98 ± 8.05 | 46.43 ± 0.05 |

The long-form CSV also contains image and pixel AUROC for every category. `all-315-runs.csv` retains unrounded source values. `main-table.tex` is a table fragment, not a report draft. `published-context-comparison.csv` contrasts our means with explicitly named paper configurations; these are descriptive gaps across implementations, not matched or significance tests. See the published-comparison note for scope.

Checks: full model/category/seed grid, saved source-file hashes, independently recomputed macro and category statistics, original fusion paired effects, and agreement between paired differences and macro differences. No image, mask, model, or reserved input is opened by this exporter.
