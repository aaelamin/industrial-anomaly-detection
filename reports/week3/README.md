# Week 3: get a baseline we can compare against

[All weeks](../README.md) | [Full results](week3.md) | [Heatmaps](HEATMAPS.md)

## What we needed to do

Get FastFlow running on the full original MVTec AD dataset and measure detection and localization. This gives later experiments a starting point. We used all 15 categories, with a separate model for each category, and repeated each run with three seeds.

MVTec AD contains normal products and textures, plus test images with defects such as scratches, cracks and contamination. Our local inventory contains 3,629 training images and 1,725 test images. Training uses only normal images. The test masks identify defective pixels and are only used for evaluation. Images are resized to 256 by 256 for the model. The original files are preserved.

## What we built and why

FastFlow uses a frozen pretrained network to extract features, then trains normalizing flows on those features. A flow learns a transformation that makes normal training features resemble a simple reference distribution. The resulting anomaly map assigns higher scores to features that appear unusual. This is not training a supervised defect classifier.

We started with ResNet-18 because it is one of the FastFlow paper's reference backbones and is relatively small. We used Anomalib 2.6.2 to provide the model and loss, with our own scripts for the fixed training schedule, checkpointing and evaluation. This made the starting implementation explicit and repeatable. It does not make our pipeline identical to the authors' experiments.

The frozen network supplies three feature levels. Each has its own flow, and the three resized output maps are averaged. The highest pixel in that combined map becomes the image score. We retained that implementation rule throughout the main experiments.

| Choice | What we used | Reason or limitation |
| --- | --- | --- |
| Feature extractor | ResNet-18 with fixed pretrained weights | A smaller reference configuration |
| Image size | 256 by 256 | The CNN input size in the FastFlow setup |
| Flow | Eight blocks, all 3 by 3 subnet convolutions | The ResNet-18 convolution variant used in the reference comparison |
| Training | Adam, learning rate 0.001, weight decay 0.00001, batch 32, 500 epochs | Fixed schedule across categories |
| Repetitions | Seeds 0, 1 and 2 | Show variation rather than a single favorable run |
| Checkpoint | Final epoch | Avoid choosing the best test checkpoint |
| Augmentation | None | Allows frozen features to be cached, but differs from the paper's augmented training |

The microbatch was eight, accumulated to an effective batch of 32. This is a memory setting, not a change to the intended batch size. A shuffled incomplete final batch is dropped each epoch. We did not select a smaller dataset, although this means every image need not participate in every epoch.

The configuration is in [week3.json](../../configs/week3.json), and the [protocol](../../docs/WEEK3_PROTOCOL.md) records the details. [FastFlow](https://arxiv.org/abs/2111.07677) is the method reference. Our implementation is [Anomalib](https://github.com/open-edge-platform/anomalib), rather than a claim of exact author-code reproduction.

## What we got

| Metric | Our mean ± seed std | Paper ResNet-18 reference |
| --- | ---: | ---: |
| Image AUROC | 90.17 ± 0.99% | 97.9% |
| Pixel AUROC | 95.41 ± 0.59% | 97.2% |
| Pixel AP | 39.69 ± 1.04% | Not compared here |

AUROC measures ranking, not thresholded accuracy. Image AUROC asks whether defective images rank above normal images. Pixel AUROC and AP measure defect localization. Each seed is averaged equally across the 15 categories before computing the table's mean and standard deviation.

Bottle and leather had very high image AUROC. Hazelnut was much weaker at 59.25%, and screw at 69.90%. A good overall pixel AUROC did not mean these categories were solved. A map could highlight a defect but also have a stronger background response, making the whole-image score unreliable.

The [checks and original examples](validation.md) recomputed all 45 runs' metrics and replayed three models. Those checks support the consistency of the files and calculations. They do not establish that our training reproduces every detail of the paper.

## Where that left us

We had a functioning full-dataset baseline, with a substantial image-level gap from the paper. We could proceed with a controlled comparison while keeping that gap visible. We should not describe this as matching the published detection performance.

The next planned step was the larger WideResNet configuration. The question was whether its representation and associated FastFlow configuration would improve the weak categories. We also needed to inspect false responses and variation between seeds, rather than judging by the average alone.

## Results and examples

- [Every category](week3.md), [category CSV](per_category.csv) and [defect CSV](per_defect.csv)
- [Heatmaps for all 15 categories and all three seeds](HEATMAPS.md)
- [What the checks and difficult examples showed](validation.md)
- [Next: Week 4](../week4/README.md)

The published gallery uses a disclosed image selection, including ordinary normal images. Full local outputs are under `results/week3/<category>/seed_<n>/`. The [run guide](../../docs/RUNNING.md) explains how to reproduce or extend the experiment.
