# Checking image scores from saved Week 4 maps

We compared five fixed ways of reducing each saved map to one image score across all 15 categories and three seeds. No models were trained, and no original outputs or running-study settings were changed. This check used CPU computation only.

Top 0.1% and top 1% average the 66 and 656 highest pixels in a 256 by 256 map. The local score takes the maximum after a fixed 9 by 9 box average with reflected edges. The whole-map mean tests a much broader summary. None uses a defect mask, per-image normalization or category-specific settings.

| Image score | Mean image AUROC | Seed std | Change from maximum |
| --- | ---: | ---: | ---: |
| maximum | 93.17% | 2.14 | +0.00 points |
| top_0.1_percent_mean | 94.03% | 1.96 | +0.86 points |
| top_1_percent_mean | 96.34% | 1.20 | +3.17 points |
| whole_map_mean | 87.02% | 0.57 | -6.15 points |
| local_9x9_mean_maximum | 93.22% | 2.08 | +0.05 points |

Each seed is first averaged equally across the 15 categories. The table reports the mean and sample standard deviation of those three averages.

## Category means across three seeds

| Category | Maximum | Top 0.1% | Top 1% | Whole map | Local 9 by 9 |
| --- | ---: | ---: | ---: | ---: | ---: |
| bottle | 99.97% | 100.00% | 100.00% | 92.62% | 100.00% |
| cable | 92.80% | 94.55% | 96.11% | 84.08% | 93.87% |
| capsule | 94.99% | 95.55% | 96.72% | 84.56% | 94.58% |
| carpet | 94.14% | 94.80% | 96.48% | 71.00% | 93.45% |
| grid | 99.75% | 99.89% | 99.86% | 70.84% | 99.92% |
| hazelnut | 78.19% | 78.82% | 89.79% | 82.39% | 76.87% |
| leather | 99.22% | 99.67% | 99.72% | 95.29% | 99.35% |
| metal_nut | 96.90% | 97.31% | 98.58% | 86.48% | 96.43% |
| pill | 89.03% | 91.67% | 94.58% | 89.70% | 91.42% |
| screw | 75.27% | 77.39% | 83.34% | 70.72% | 73.68% |
| tile | 99.58% | 99.81% | 99.96% | 99.34% | 99.83% |
| toothbrush | 88.33% | 89.07% | 94.44% | 94.17% | 88.89% |
| transistor | 93.39% | 95.60% | 98.31% | 90.39% | 94.04% |
| wood | 97.84% | 97.95% | 98.71% | 95.61% | 97.98% |
| zipper | 98.16% | 98.32% | 98.56% | 98.17% | 97.95% |

All 45 saved map, score and metric file hashes passed verification. Maximum scores matched the saved scores exactly and reproduced their image AUROCs. Image order and labels were checked. Pixel metrics are unchanged because the pixel maps themselves are unchanged.

This is an exploratory sensitivity check on an already inspected test set. Better scores here cannot be presented as an independently validated improvement or as reproduction of the authors' image-scoring rule. We report every method and category rather than picking a different winner for each category. Any chosen change needs a fixed protocol and independent confirmation.

- [Every run](image_scores/per_run.csv)
- [Every defect type](image_scores/per_defect.csv)
- [Image scores](image_scores/image_scores.csv)
- [Checks and summary](image_scores/summary.json)
