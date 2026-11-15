# Methods and implementation

The study uses an Anomalib-based FastFlow adaptation. The details below describe the tested implementation; the paper’s headline scores were not exactly reproduced. Final numbers are in the complete result tables.

## Data and evaluation units

MVTec AD uses its official normal-only training and mixed normal/anomalous test split: 3,629 training images and 1,725 test images over 15 categories. Each configuration is repeated with seeds 0, 1 and 2. The same test images recur across seeds; pixels and seeds do not establish independent physical-object replication. Input images are 256 × 256 with ImageNet normalization. Released defect masks use nearest-neighbor resizing.

Each category's pixel metrics pool its test pixels. Image AUROC ranks normal versus anomalous images; pixel AUROC ranks defect versus nondefect pixels; pixel AP measures precision-recall ranking on the fixed defect prevalence. Average the 15 category scores equally within each seed, then report the mean and **sample** SD of those three seed means. Do not pool all categories' pixels for the reported category-macro score. AUROC and AP are not thresholded accuracy or a probability of a defect.

The separate AeBAD experiment fits a new flow on 128 official-training normal images, calibrates on 32 separate training normals, and evaluates 160 declared development images: four capture-condition groups × five labels × eight images. All three seeds use those same images. The provider defines changed test backgrounds, lighting and viewpoints; repeated folder names do not establish matching capture settings. Physical-blade and capture-session independence remain unverified. Learned MVTec flow weights are not transferred. The 1,479 reserved AeBAD images remain excluded.

## Flow objective and displayed score

For a normalized backbone feature tensor $h_l$ and invertible flow $f_l$, write $z_l=f_l(h_l)$. The implemented training objective, up to constants, is

$$\mathcal L = \mathbb E_{x\sim D_{\mathrm{normal}}}\sum_l\left[\tfrac12\|z_l\|_2^2-\log|\det J_{f_l}(h_l)|\right].$$

The feature extractor is frozen. This is a density objective in the representation supplied to the flow, not an exact density of the original RGB image. The trainable normalization preceding the flow is another reason to avoid interpreting the objective as an end-to-end invertible image density.

The saved anomalib localization map is built from latent energy, with no spatial Jacobian term in that map:

$$a_l(u,v)=-\exp\!\left[-\frac{1}{2C_l}\sum_c z_l(c,u,v)^2\right],\qquad A(x)=\frac1L\sum_l\operatorname{Bilinear}_{256}(a_l).$$

The image score is the maximum of the saved map. Larger values, closer to zero, are more anomalous. This score is not a calibrated defect probability. Each gallery heatmap has an independent display range; compare locations with the mask, not brightness between models. Nondefect pixels include intact object regions as well as physical background.

## Compact multi-scale extension and matched ablation

The frozen WideResNet-50-2 extractor supplies three feature levels at strides 4, 8 and 16. For each level, normal training patches alone fit a randomized PCA projection. At most 50,000 patches are sampled with projection seed 2026; four power iterations are used. The middle level stores 96 components, and the outer levels 32 each. Retained components are centered and scaled by their training projection standard deviations, with a $10^{-6}$ floor.

The middle-only control takes 96 components from the middle level. The fused arm takes 32 components from each of the three levels and concatenates them. Both resize projected features bilinearly to a 64 × 64 grid and have 96 channels. Both then use a trainable LayerNorm and the same eight-block flow, with 1,065,600 trainable parameters. Paired runs share initial flow parameters and data order. The contrast tests this particular feature-level allocation at a fixed compact representation and flow capacity.

```text
Fit, using normal training images only:
    extract frozen three-level features
    fit PCA and component scales at each level
For each seed, initialize one shared compact-flow state:
    middle = resize(project_middle(features_middle, 96), 64x64)
    fused  = concat(resize(project_level(features_level, 32), 64x64)
                    for each of the three levels)
    train a separate copy of that flow state for each arm to epoch 500
    evaluate both saved endpoints on the same declared test images
```

This is not a claim to invent multi-scale flow modeling. Compared with the larger WideResNet reference, compression, normalization, spatial resizing and the number of flows change together. The large reference-to-compact gap therefore cannot be assigned to fusion alone. The middle-versus-fusion pair is the relevant ablation.

## Reference recipes

| Configuration | Pretrained weight SHA prefix | Coupling kernels | Microbatch |
|---|---|---|---:|
| Original ResNet-18 | `5c106cde386e` | all 3 × 3 | 8 |
| Matched ResNet-18 | `5c106cde386e` | alternating 3 × 3 / 1 × 1 | 8 |
| WideResNet-50-2 | `8234f177589d` | alternating | 4 |
| Compact middle and fusion | `8234f177589d` | alternating | 32 |
| EfficientNet-B4 | `7eb33cd5c820` | alternating | 32 |
| AeBAD ResNet-18 | `5c106cde386e` | all 3 × 3 | 4 |

All flow endpoints are fixed at 500 epochs: Adam, learning rate 0.001, weight decay $10^{-5}$, effective batch 32, no augmentation or scheduler. The added reference flows use eight blocks per level and hidden ratio one. Effective batches use microbatch gradient accumulation where necessary; incomplete final batches are dropped. The added runs use deterministic FP32, disabled TF32 and four CPU threads. The recipe JSON contains complete identities and optimizer defaults.

The original and matched ResNet recipes share pretrained weights and microbatch settings. The matched route addresses the kernel-pattern mismatch with WideResNet, but execution routes also differ; avoid an unrestricted claim that all differences are caused by one architectural factor. EfficientNet weights were pretrained at 320 pixels and are applied at the shared 256-pixel input. That is an adaptation, and its weak scores do not establish a general ranking of backbone quality.

PatchCore is the second published-method family. Its retained implementation uses nominal patch features and a 1% coreset; its seeds vary coreset sampling rather than neural optimization. The exported PatchCore recipe and source specify the layers and scoring. Its run-to-run variation comes from reference selection, unlike the optimization randomness in flow training.

## AeBAD operating point and numerical checks

The native localization grid is 3024 × 3024. The frozen evaluator adds float32 +1 to the flow map before bilinear interpolation and uses $2^{20}$ histogram bins on [0,2]. A normal-only 99th-percentile calibration rule is applied separately at initialization and epoch 500. Defect labels do not select the threshold. Initialization is a diagnostic control, not a competitive baseline.

Released-mask regions use 8-connectivity. A small region occupies at most 0.1% of image area; a detected region has at least 10% of its pixels above the calibrated threshold. The small and fracture subsets overlap. Report previously detected regions lost **and** newly detected regions gained alongside normal-pixel false-positive rate, AP and PRO at a 5% false-positive allowance. Counts alone can improve when more normal pixels are flagged.

The saved-metric verifier checks the same region population across endpoints and seeds, gains minus losses against net detection changes, aggregation and numerical ranking bounds. Those bounds quantify approximation error, not a confidence interval. The saved-number verifier does not independently rerun inference or all pixel-metric construction.

## Operational provenance

A laptop RTX 4050 with 6 GiB VRAM and 16 GB system RAM ran these experiments. Checkpoints, source hashes and allowlisted access receipts were retained. Resource pauses kept the fixed scientific endpoint. A qualified memory-mapped feature-cache and atomic-write recovery layer reduced operational failures; it is an implementation safeguard, not an accuracy contribution.

The combined reproduction suite materializes original source files into fresh output folders and regenerates recovery prerequisites. Its status receipt states the actual qualification achieved. A same-machine short recovery test does not establish a clean installation on another machine or a complete second execution of all 318 jobs.

Sources: [FastFlow](https://arxiv.org/abs/2111.07677), [Real NVP](https://arxiv.org/abs/1605.08803), [PatchCore](https://arxiv.org/abs/2106.08265), [AeBAD](https://arxiv.org/html/2304.02216v2). Implementation details above were checked against the frozen local sources; they are not all claims made by those papers.
