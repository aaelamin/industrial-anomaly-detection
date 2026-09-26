# Week 3: FastFlow on MVTec AD

Status: complete (45/45 runs).

ResNet-18; 500 epochs; effective batch size 32; seeds 0, 1 and 2. Training uses only normal images. All results use the final checkpoint.

| Category | Runs | Image AUROC (%) | Pixel AUROC (%) | Pixel AP (%) |
| --- | ---: | ---: | ---: | ---: |
| bottle | 3 | 99.95 ± 0.09 | 98.06 ± 0.18 | 67.64 ± 1.76 |
| cable | 3 | 85.79 ± 0.56 | 94.05 ± 0.93 | 31.00 ± 3.99 |
| capsule | 3 | 88.63 ± 3.69 | 97.43 ± 0.19 | 26.97 ± 1.13 |
| carpet | 3 | 98.98 ± 1.07 | 98.88 ± 0.08 | 63.05 ± 1.47 |
| grid | 3 | 98.55 ± 0.39 | 97.96 ± 0.24 | 37.22 ± 3.24 |
| hazelnut | 3 | 59.25 ± 5.41 | 93.16 ± 1.29 | 23.82 ± 1.15 |
| leather | 3 | 100.00 ± 0.00 | 99.37 ± 0.04 | 45.33 ± 4.69 |
| metal_nut | 3 | 87.80 ± 4.81 | 89.05 ± 6.74 | 55.23 ± 9.23 |
| pill | 3 | 91.53 ± 1.71 | 94.75 ± 0.33 | 38.28 ± 3.24 |
| screw | 3 | 69.90 ± 3.28 | 94.19 ± 0.43 | 5.56 ± 1.16 |
| tile | 3 | 98.61 ± 0.47 | 92.88 ± 0.32 | 44.19 ± 1.91 |
| toothbrush | 3 | 87.87 ± 0.16 | 96.93 ± 0.19 | 24.34 ± 1.30 |
| transistor | 3 | 93.08 ± 3.18 | 96.43 ± 0.92 | 58.87 ± 6.68 |
| wood | 3 | 98.25 ± 0.09 | 90.64 ± 1.09 | 39.25 ± 2.33 |
| zipper | 3 | 94.32 ± 3.31 | 97.34 ± 0.67 | 34.66 ± 4.54 |

## Overall results

- Image AUROC: 90.17 ± 0.99%. Paper ResNet-18 reference: 97.9%; difference -7.73 percentage points.
- Pixel AUROC: 95.41 ± 0.59%. Paper ResNet-18 reference: 97.2%; difference -1.79 percentage points.

Each seed contributes one mean across all 15 categories. The reported standard deviation is across those three means.

## Cases to inspect

The table below ranks test images by their average pairwise ranking error. For an anomaly, this is the fraction of normal images that receive a higher score. For a normal image, it is the fraction of anomalies that receive a lower score. Ties count as one half. No decision threshold is chosen from the test set.

| Category | Image | Mean ranking error | Seeds with errors |
| --- | --- | ---: | ---: |
| hazelnut | test\good\001.png | 0.967 | 3/3 |
| screw | test\good\014.png | 0.944 | 3/3 |
| hazelnut | test\good\039.png | 0.938 | 3/3 |
| screw | test\scratch_neck\023.png | 0.935 | 3/3 |
| screw | test\good\025.png | 0.933 | 3/3 |
| screw | test\good\003.png | 0.922 | 3/3 |
| screw | test\good\013.png | 0.913 | 3/3 |
| hazelnut | test\good\011.png | 0.871 | 3/3 |
| hazelnut | test\good\007.png | 0.862 | 3/3 |
| cable | test\missing_wire\000.png | 0.845 | 3/3 |
| screw | test\scratch_head\004.png | 0.829 | 3/3 |
| hazelnut | test\good\010.png | 0.829 | 3/3 |
| hazelnut | test\good\022.png | 0.829 | 3/3 |
| hazelnut | test\good\002.png | 0.824 | 3/3 |
| screw | test\thread_side\021.png | 0.821 | 3/3 |

## Comparison limits

This uses Anomalib and the older reference ImageNet weights, not the authors’ original code. Pixel metrics use 256 x 256 binary masks. The paper’s 99.4% headline and main per-category table use another backbone and are not matched targets for this study. The reference implementation uses no augmentation; this study follows that choice.

Defect-level tables compare each defect type with the normal images in its category. Small test groups can produce unstable estimates. Test-set failure analysis is descriptive; it does not establish a cause or justify tuning on these examples.

Source: [FastFlow](https://arxiv.org/html/2111.07677v2), Tables 1 and 5. See `docs/WEEK3_PROTOCOL.md` for the fixed settings and implementation choices.
