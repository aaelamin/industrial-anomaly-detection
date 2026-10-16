# Looking at the three feature levels

These are inference-only output ablations of the saved Week 4 models. The three-level average is the original baseline. No model was trained or selected using these scores. Every category uses all three seeds. These five categories were selected after inspecting the earlier test results, so this is exploratory.

| Category | Levels | Image AUROC | Pixel AUROC | Pixel AP |
| --- | --- | ---: | ---: | ---: |
| bottle | 1 | 99.66 ± 0.05% | 98.09 ± 0.12% | 77.12 ± 1.82% |
| bottle | 2 | 99.11 ± 1.40% | 97.32 ± 0.92% | 60.02 ± 5.37% |
| bottle | 3 | 99.84 ± 0.27% | 95.78 ± 0.55% | 45.00 ± 2.41% |
| bottle | 1+2 | 100.00 ± 0.00% | 98.70 ± 0.12% | 76.36 ± 1.85% |
| bottle | 1+3 | 99.97 ± 0.05% | 98.78 ± 0.11% | 78.65 ± 1.40% |
| bottle | 2+3 | 100.00 ± 0.00% | 97.75 ± 0.21% | 61.14 ± 1.74% |
| bottle | 1+2+3 | 99.97 ± 0.05% | 98.84 ± 0.06% | 77.05 ± 1.74% |
| screw | 1 | 48.88 ± 4.54% | 76.11 ± 23.43% | 1.55 ± 0.93% |
| screw | 2 | 67.81 ± 6.16% | 93.75 ± 2.22% | 10.88 ± 0.79% |
| screw | 3 | 72.99 ± 9.74% | 93.46 ± 1.97% | 7.51 ± 3.37% |
| screw | 1+2 | 65.72 ± 5.71% | 94.47 ± 1.67% | 9.26 ± 0.55% |
| screw | 1+3 | 66.37 ± 11.67% | 94.62 ± 0.22% | 6.81 ± 2.58% |
| screw | 2+3 | 79.65 ± 6.16% | 96.19 ± 1.59% | 16.09 ± 2.54% |
| screw | 1+2+3 | 75.27 ± 9.04% | 96.87 ± 0.44% | 14.22 ± 2.98% |
| hazelnut | 1 | 65.01 ± 1.82% | 87.21 ± 2.92% | 36.42 ± 0.37% |
| hazelnut | 2 | 77.02 ± 4.86% | 85.34 ± 5.00% | 22.43 ± 2.28% |
| hazelnut | 3 | 92.24 ± 1.74% | 96.23 ± 1.03% | 27.06 ± 4.85% |
| hazelnut | 1+2 | 72.07 ± 1.79% | 89.41 ± 2.85% | 35.25 ± 1.86% |
| hazelnut | 1+3 | 80.38 ± 4.57% | 96.21 ± 1.63% | 45.65 ± 2.61% |
| hazelnut | 2+3 | 87.05 ± 5.01% | 94.51 ± 1.86% | 29.06 ± 4.84% |
| hazelnut | 1+2+3 | 78.19 ± 2.53% | 95.52 ± 1.80% | 40.06 ± 3.26% |
| carpet | 1 | 64.38 ± 18.71% | 78.21 ± 12.12% | 23.72 ± 21.20% |
| carpet | 2 | 90.42 ± 2.27% | 97.73 ± 0.53% | 44.12 ± 8.90% |
| carpet | 3 | 91.96 ± 5.51% | 96.35 ± 0.97% | 21.79 ± 7.16% |
| carpet | 1+2 | 89.78 ± 11.72% | 97.35 ± 1.01% | 54.93 ± 8.96% |
| carpet | 1+3 | 82.09 ± 18.94% | 96.02 ± 1.60% | 41.22 ± 17.48% |
| carpet | 2+3 | 96.70 ± 0.81% | 98.38 ± 0.29% | 44.56 ± 8.03% |
| carpet | 1+2+3 | 94.14 ± 6.89% | 98.49 ± 0.31% | 59.43 ± 4.59% |
| pill | 1 | 86.97 ± 2.39% | 91.78 ± 4.07% | 33.82 ± 4.02% |
| pill | 2 | 79.61 ± 10.79% | 67.18 ± 40.06% | 24.20 ± 19.89% |
| pill | 3 | 91.37 ± 3.61% | 94.40 ± 0.85% | 40.46 ± 3.00% |
| pill | 1+2 | 84.75 ± 14.43% | 87.95 ± 9.71% | 34.93 ± 19.42% |
| pill | 1+3 | 93.74 ± 1.14% | 96.10 ± 0.25% | 47.12 ± 9.92% |
| pill | 2+3 | 86.42 ± 14.85% | 82.84 ± 22.15% | 34.06 ± 26.72% |
| pill | 1+2+3 | 89.03 ± 10.80% | 93.67 ± 4.67% | 42.73 ± 20.95% |

Levels 1, 2 and 3 have native sizes 64 x 64, 32 x 32 and 16 x 16. Subsets use equal averaging after bilinear resizing to 256 x 256. Image scoring always uses the maximum. Standard deviations are across the three seeds. These are output ablations of jointly trained independent branches, not independently trained single-level models.

All 15 extracted three-level averages matched the model map generator exactly and passed comparison with the saved maps (rtol 1e-5, atol 1e-6). [Exact differences](level_checks.json) and [all seed results](level_subsets.csv) are saved. [Level ranges](level_ranges.csv) describe normal test maps and must not be used as training calibration.

Do not pick a separate best subset for each category from this table. The feature-fusion pilot uses the design fixed in the [Week 5 plan](../../docs/WEEK5_PLAN.md).
