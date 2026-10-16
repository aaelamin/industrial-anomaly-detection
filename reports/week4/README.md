# Week 4: compare the larger FastFlow configuration

[All weeks](../README.md) | [Full results](week4.md) | [Heatmaps](HEATMAPS.md)

## What we needed to do

Compare the ResNet-18 baseline with WideResNet-50-2. We repeated all 15 categories with the same three seeds, normal-only training split, 256 by 256 input and final-epoch evaluation after 500 epochs.

The broader proposal also mentions EfficientNet-B4. It was not run here. This report covers the completed ResNet-18 versus WideResNet comparison.

## What changed and why

WideResNet-50-2 is a deeper network than ResNet-18 and uses wider bottleneck blocks. We kept its pretrained backbone frozen, so we were training the flows, not relearning the backbone from the small MVTec training sets. It supplies a richer but more expensive feature representation.

We used the WideResNet reference weights and alternating 3 by 3 and 1 by 1 flow subnet convolutions. The latter follows the FastFlow paper's WideResNet configuration. Week 3 used all 3 by 3 convolutions. Therefore this is a comparison of reference configurations, not a clean experiment changing only backbone depth or width. The [configuration](../../configs/week4.json) identifies the exact weights and settings.

The effective batch stayed at 32. The microbatch dropped from eight to four to fit GPU memory. Frozen features were cached on disk, and recovery checks established that interrupted training could continue with the same model, optimizer and random states. This made the longer runs manageable. It did not remove the no-augmentation difference from the paper.

## What we got

| Metric | Week 3 | Week 4 | Change |
| --- | ---: | ---: | ---: |
| Image AUROC | 90.17 ± 0.99% | 93.17 ± 2.14% | +3.00 points |
| Pixel AUROC | 95.41 ± 0.59% | 97.19 ± 0.49% | +1.78 points |
| Pixel AP | 39.69 ± 1.04% | 52.99 ± 2.95% | +13.29 points |

These are category-balanced averages, with standard deviations across three seed averages. The configuration improved overall detection and localization. It did not help every category.

Hazelnut image AUROC improved from 59.25% to 78.19%. Screw improved from 69.90% to 75.27%, with one seed getting worse. Carpet and pill declined on average. The [full table](week4.md) includes every category, not just the gains.

## What the closer inspection showed

The [loss and heatmap analysis](followup.md) found several different issues. Some hazelnut and screw defects became easier to locate, but strong background or border responses remained. Pill seed 0 had a sharp upward loss jump near the end, at epoch 497, and substantially weaker results than its other seeds. This suggests a stability question, but does not prove the spike caused the poor test result.

A separate [image-score check](image_score_check.md) used the saved maps without retraining. The original maximum score lets one pixel determine the image ranking. Averaging a small group of high-scoring pixels tests whether a response has some spatial support while preserving sensitivity to small defects.

Averaging the highest 1% increased image AUROC from 93.17% to 96.34%. The report also includes the highest 0.1%, the whole-map average and a local-average alternative. The exact 1% choice was not established as optimal in advance. This is exploratory evidence about scoring sensitivity, not a replacement for the baseline or an independently validated improvement. Pixel metrics stay unchanged because the maps stay unchanged.

## Where that left us

Week 4 was our strongest tested configuration. Its 97.19% pixel AUROC was close to the paper's 98.1% reference, while image AUROC remained 6.13 points below the 99.3% reference. The [FastFlow paper](https://arxiv.org/abs/2111.07677) uses augmentation, but we have not tested it and cannot claim it explains this gap.

Before Week 5, we exposed the individual feature-level maps on five inspected categories. Different levels behaved differently, so equal averaging could hide useful or harmful contributions. That motivated asking whether a flow could learn from the levels together. It did not establish that combining them would help, or that the chosen compact implementation was the best way to do it.

The next experiment therefore needed a matched control and explicit reporting of failures. Training stability and the baseline reproduction gap remained separate questions.

## Results and examples

- [Every category and comparison with Week 3](week4.md), [category CSV](per_category.csv) and [defect CSV](per_defect.csv)
- [Heatmaps for all 15 categories](HEATMAPS.md)
- [Matched examples and loss histories](followup.md)
- [All image-scoring alternatives](image_score_check.md)
- [Next: Week 5](../week5/README.md)

Full local results are under `results/week4/runs/<category>/seed_<n>/`. The older preflight and Colab reports record setup decisions, not later study status. See the [run guide](../../docs/RUNNING.md) for a fresh-machine route that does not depend on the original local recovery directory.
