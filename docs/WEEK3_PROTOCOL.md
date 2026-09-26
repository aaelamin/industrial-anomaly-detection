# Week 3 experiment

The first study attempts to reproduce the ResNet-18 FastFlow baseline on all 15 MVTec AD categories. Settings were fixed before evaluating the full study. The completed results and accuracy gap are documented in [the validation report](../reports/week3/validation.md).

| Setting | Value |
| --- | --- |
| Backbone | ResNet-18, frozen |
| Weights | `resnet18-5c106cde.pth`, the timm 0.5.4 reference weights |
| Features | Last outputs of layer1, layer2, layer3 |
| Input | 256 x 256, ImageNet normalization |
| Flows | Eight steps per feature level; all 3 x 3 convolutions |
| Hidden ratio | 1.0 |
| Optimizer | Adam, learning rate 0.001, weight decay 0.00001 |
| Schedule | 500 epochs, no early stopping |
| Batch | 32, incomplete training batches dropped after shuffling |
| Seeds | 0, 1, 2 |
| Augmentation | None |
| Checkpoint used for evaluation | Final epoch |

Sources: [FastFlow, sections 4.4 and 4.7](https://arxiv.org/html/2111.07677v2), [reference configuration](https://github.com/gathierry/FastFlow/blob/master/configs/resnet18.yaml), [reference training code](https://github.com/gathierry/FastFlow/blob/master/main.py), and [timm 0.5.4 weights](https://github.com/huggingface/pytorch-image-models/blob/v0.5.4/timm/models/resnet.py).

## Implementation choices

The flow model and loss come from Anomalib 2.6.2, which credits the unofficial gathierry implementation. This is not the paper authors' original training code. The exact comparison therefore has limits.

The frozen backbone's outputs are cached in RAM. Layer normalization and flows still train on every batch. A batch of 32 is processed as four microbatches of eight; gradients are averaged before one optimizer update. A test checks the cached loss and accumulated gradients against a direct full-batch calculation. Computation is float32 with deterministic algorithms enabled.

The reference implementation uses no data augmentation. The paper's supplement describes augmentation but does not fully specify category rules or rotation ranges. We follow the inspectable reference implementation and report that choice.

## Evaluation

Each category has its own trained model. All official test images are evaluated, and labels are never used to choose an epoch or training setting. Image AUROC uses the maximum anomaly-map value. Pixel AUROC and average precision pool pixels within each category at 256 x 256; binary masks use nearest-neighbor resizing. This mask handling is explicit because the older reference loader resized masks bilinearly.

The paper's ResNet-18 aggregate is 97.9% image AUROC and 97.2% pixel AUROC. Its headline 99.4% image AUROC and detailed main table use a stronger backbone, so they are not matched per-category targets for this study.

Results are reported as per-category mean and sample standard deviation across seeds. The overall figure is the mean of the 15 category AUROCs, computed separately for each seed, then summarized across the three seeds. Scores from different category models are not pooled into one ROC curve.

## Research follow-up

Save every test score and anomaly map. Compare defect types against the normal images within the same category. Inspect the highest-scoring normal images and lowest-scoring anomalies, and check whether the same images are difficult across seeds. These are descriptive test-set analyses, not model-selection data or proof of a failure mechanism.

The next intervention remains undecided. Any improvement suggested by this study needs separate validation; repeatedly tuning on these test examples would make the final comparison optimistic.
