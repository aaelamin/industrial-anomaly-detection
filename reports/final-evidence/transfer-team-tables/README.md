# AeBAD development tables for the report team

Three seeds on the same declared 160 development images. Initialization is a diagnostic control, not a competitive method. Mean and sample SD describe seed variation. Macro scores weight the four capture-condition groups equally within each seed.

| Metric | Initialization (%) | Epoch 500 (%) | Paired change (points) |
|---|---:|---:|---:|
| Image AUROC | 50.23 ± 0.74 | 44.24 ± 5.26 | -5.99 ± 4.62 |
| Pixel AUROC | 84.50 ± 0.05 | 71.97 ± 2.65 | -12.53 ± 2.60 |
| Pixel AP | 3.10 ± 0.01 | 2.30 ± 0.50 | -0.80 ± 0.49 |
| PRO (5% FPR) | 29.31 ± 0.37 | 14.66 ± 0.41 | -14.66 ± 0.75 |
| Normal-pixel FPR | 0.78 ± 0.02 | 13.62 ± 2.56 | 12.85 ± 2.58 |

Higher is better for ranking metrics; lower is better for false-positive rate. Pixel AP and PRO use the frozen native-grid histogram evaluation. Numerical approximation bounds are in the verification receipt and are not statistical confidence intervals.

## Epoch-500 capture-condition breakdown

| Provider group | Image AUROC (%) | Pixel AUROC (%) | Pixel AP (%) | PRO (5% FPR) (%) | Normal-pixel FPR (%) |
|---|---:|---:|---:|---:|---:|
| same | 48.96 ± 10.53 | 86.10 ± 4.19 | 3.60 ± 0.46 | 32.40 ± 3.13 | 1.24 ± 0.03 |
| background | 42.32 ± 5.67 | 47.59 ± 3.49 | 0.79 ± 0.09 | 2.37 ± 3.66 | 39.99 ± 15.39 |
| illumination | 45.31 ± 6.50 | 75.03 ± 8.87 | 2.51 ± 1.62 | 8.62 ± 5.51 | 8.18 ± 4.29 |
| view | 40.36 ± 10.56 | 79.16 ± 1.75 | 2.29 ± 0.38 | 15.24 ± 1.00 | 5.08 ± 1.53 |

These are provider folder names. In particular, `same` is not a certificate of matched train/test capture conditions or physical-object independence.

## Region transitions at separately calibrated normal-only thresholds

| Subset | Regions per seed | Initially detected (seeds 0 / 1 / 2) | Finally detected | Lost | Gained |
|---|---:|---|---|---|---|
| all | 240 | 57 / 56 / 60 | 98 / 91 / 98 | 19 / 19 / 18 | 60 / 54 / 56 |
| small | 144 | 31 / 31 / 35 | 54 / 55 / 58 | 11 / 9 / 9 | 34 / 33 / 32 |
| fracture | 32 | 9 / 8 / 8 | 9 / 13 / 13 | 5 / 2 / 1 | 5 / 7 / 6 |

Small means at most 0.1% of native image area. A region is detected when at least 10% of its pixels exceed the normal-only calibrated threshold. Small and fracture subsets overlap. Gains do not erase losses; net gains must be interpreted alongside the increased normal-pixel false-positive rate. Nondefect mask pixels include intact objects as well as physical background.

The method is refitted on 128 new-domain training normals and calibrated on 32 separate training normals. This is a domain application, not zero-shot transfer of learned MVTec flow weights. Dataset-path separation is verified; physical-blade and acquisition-session independence are not. The 1,479 reserved AeBAD images remain excluded.

The CSVs retain unrounded values, all seed/condition endpoints, region transitions and the recorded calibration thresholds. The exporter loads only saved JSON/CSV evidence and runs its standard-library verifier. It performs no inference, metric rescoring or threshold selection.
