# Week 4: FastFlow with WideResNet-50

We repeated the MVTec AD experiment with WideResNet-50-2 as the feature extractor. All 15 categories used seeds 0, 1 and 2, with 500 epochs per run. Training used normal images only. Evaluation used the final checkpoint and the same test images as Week 3.

## What changed

Week 3 used ResNet-18. This experiment used WideResNet-50-2 and its reference pretrained weights. It also used alternating 3x3 and 1x1 flow convolutions instead of all 3x3. The microbatch changed from 8 to 4 to fit memory, while the effective batch stayed at 32. Both experiments used 256 x 256 images, eight flow steps, the same learning rate and no augmentation.

This compares two FastFlow configurations. It does not isolate the effect of backbone size, because the pretrained weights and flow convolutions also changed.

## Results

| Metric | Week 3, ResNet-18 | Week 4, WideResNet-50 | Change (points) |
| --- | ---: | ---: | ---: |
| Image AUROC | 90.17 ± 0.99% | 93.17 ± 2.14% | +3.00 |
| Pixel AUROC | 95.41 ± 0.59% | 97.19 ± 0.49% | +1.78 |
| Pixel AP | 39.69 ± 1.04% | 52.99 ± 2.95% | +13.29 |

Each seed contributes an average across the 15 categories. The table shows the mean and sample standard deviation of those three averages. These are ranking metrics, not percentages of images classified correctly.

| Category | Week 3 image AUROC | Week 4 image AUROC | Change (points) | Week 3 pixel AUROC | Week 4 pixel AUROC |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 99.95% | 99.97% | +0.03 | 98.06% | 98.84% |
| cable | 85.79% | 92.80% | +7.00 | 94.05% | 96.10% |
| capsule | 88.63% | 94.99% | +6.36 | 97.43% | 98.38% |
| carpet | 98.98% | 94.14% | -4.84 | 98.88% | 98.49% |
| grid | 98.55% | 99.75% | +1.20 | 97.96% | 98.97% |
| hazelnut | 59.25% | 78.19% | +18.94 | 93.16% | 95.52% |
| leather | 100.00% | 99.22% | -0.78 | 99.37% | 99.63% |
| metal_nut | 87.80% | 96.90% | +9.11 | 89.05% | 96.95% |
| pill | 91.53% | 89.03% | -2.51 | 94.75% | 93.67% |
| screw | 69.90% | 75.27% | +5.37 | 94.19% | 96.87% |
| tile | 98.61% | 99.58% | +0.97 | 92.88% | 96.59% |
| toothbrush | 87.87% | 88.33% | +0.46 | 96.93% | 97.92% |
| transistor | 93.08% | 93.39% | +0.31 | 96.43% | 96.60% |
| wood | 98.25% | 97.84% | -0.41 | 90.64% | 94.85% |
| zipper | 94.32% | 98.16% | +3.84 | 97.34% | 98.44% |

## What happened to screw and hazelnut?

Both improved on average, but neither caught up with the stronger categories. Hazelnut improved in all three corresponding seeds. Screw improved in seeds 1 and 2, while seed 0 got worse. The larger configuration helped, but it did not give a consistent fix for screw.

| Category | Seed | Week 3 image AUROC | Week 4 image AUROC | Week 4 pixel AP |
| --- | ---: | ---: | ---: | ---: |
| hazelnut | 0 | 65.50% | 76.54% | 37.74% |
| hazelnut | 1 | 56.04% | 81.11% | 43.78% |
| hazelnut | 2 | 56.21% | 76.93% | 38.65% |
| screw | 0 | 67.88% | 64.99% | 10.79% |
| screw | 1 | 73.68% | 78.79% | 15.76% |
| screw | 2 | 68.13% | 82.03% | 16.11% |

Screw still has low pixel AP despite its high pixel AUROC. That means we should not describe its heatmaps as precise defect masks. These metrics measure different things, and the defect covers only a small part of many screw images.

We repeated the Week 3 check of where each map reaches its highest value. These are averages across the three seeds.

| Category | Week 3 normal peaks at border | Week 4 normal peaks at border | Week 3 defect peaks inside mask | Week 4 defect peaks inside mask |
| --- | ---: | ---: | ---: | ---: |
| bottle | 8.33% | 6.67% | 85.71% | 88.89% |
| hazelnut | 55.00% | 47.50% | 52.38% | 62.38% |
| screw | 57.72% | 55.28% | 13.73% | 22.69% |

Border means the outer 16 pixels of the 256 x 256 map, as in the Week 3 check. It is only a rough location check, not a background segmentation or proof of the cause.

## What to do next

The [follow-up comparison](followup.md) examines the loss curves and matched images. It found a late loss spike in pill seed 0 and persistent background responses in selected screw and hazelnut images. The [Week 5 plan](../../docs/WEEK5_PLAN.md) proposes inspecting the individual feature levels before changing their fusion.

Keep all three seeds and use these results for the Week 4 comparison. Carpet and pill also dropped on average, with particularly weak seed-0 runs. We should inspect those alongside screw and hazelnut rather than rerun only the weak results until they improve.

Week 5 is the multi-scale flow implementation. FastFlow already uses three feature levels, so the group needs to choose a specific change to how those levels are combined. Keep one configuration fixed and compare the existing map averaging with that change. Decide the settings before running, use normal training data for fitting, and avoid choosing settings from the test scores we just inspected.

If we want to claim that backbone size alone caused an improvement, we first need a comparison with the same flow convolution pattern. The experiment here supports a comparison of configurations, not that narrower claim. Reproducing the paper remains a separate unresolved issue.

## Checking the saved runs

All 45 runs passed the file checksum checks and had 500 recorded epochs with finite losses. Image scores matched the maximum of each saved anomaly map. Image AUROC, pixel AUROC, pixel AP and defect-level image AUROC were recalculated against the test labels and masks and matched the saved metrics within 1e-12. Test paths and labels matched Week 3.

The bottle, hazelnut and screw models from seed 0 were loaded again and run on their full test sets. Their maps matched the saved outputs within the replay tolerance (rtol 1e-5, atol 1e-6). Exact differences are in [checks.json](checks.json). This checks result integrity, not whether our implementation reproduces the paper.

## Files

- [Category comparison and standard deviations](per_category.csv)
- [Defect scores for each seed](per_defect.csv)
- [Overall numbers](summary.json)
- [Checks and model replay](checks.json)
- Local gallery: `results/week4/heatmaps.html`. Open it in a browser. It needs the heatmap folders beside it, which are excluded from Git.
- Full outputs: `results/week4/runs/<category>/seed_<n>/`.

To repeat these checks and rebuild the tables and gallery, run `python scripts/week4_report.py` from the repository folder. This needs both studies, the dataset and a CUDA GPU. It runs inference for three saved models and does not train.
