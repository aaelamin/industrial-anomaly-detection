# Related work

The closest work covers normalizing flows, multi-scale localization, patch-memory methods and changes in capture conditions. References were checked against primary sources on 13 November 2026.

| Paper | Connection to our work | Differences and limitations |
|---|---|---|
| Yu et al., *FastFlow: Unsupervised Anomaly Detection and Localization via 2D Normalizing Flows* (2021), [paper](https://arxiv.org/abs/2111.07677) | The baseline method family: frozen deep features and 2D flows. | Our anomalib-based recipes are adaptations. Published results are context, not directly comparable local reruns. |
| Dinh, Sohl-Dickstein and Bengio, *Density Estimation using Real NVP* (ICLR 2017), [paper](https://arxiv.org/abs/1605.08803) | Invertible transformations, coupling and change of variables. | It does not establish that every likelihood score is a reliable industrial anomaly score. |
| Gudovskiy, Ishizaka and Kozuka, *CFLOW-AD* (WACV 2022), [paper](https://arxiv.org/abs/2107.12571) | Closely related conditional flow approach to normal-only localization. | We did not run CFLOW-AD. |
| Roth et al., *Towards Total Recall in Industrial Anomaly Detection* (CVPR 2022), [paper](https://arxiv.org/abs/2106.08265) | PatchCore, the second implemented method family and a patch-memory comparator. | Our common-grid/coreset adaptation does not reproduce every published setting. |
| Zhou et al., *MSFlow: Multi-Scale Flow-based Framework for Unsupervised Anomaly Detection* (2023), [paper](https://arxiv.org/abs/2308.15300) | Parallel scale-specific flows followed by a fusion flow. | Multi-scale flow fusion is already established. Our design concatenates compressed features before one flow; it did not establish an improvement. |
| Lei et al., *PyramidFlow: High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow* (CVPR 2023), [paper](https://arxiv.org/abs/2303.02595) | High-resolution localization, pyramid flow and volume normalization. | Resolution and normalization in flow localization are already studied here. |
| Bergmann et al., *MVTec AD—A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection* (CVPR 2019), [provider and paper link](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) | Official benchmark roles and pixel annotations. | Dataset images and derived gallery examples retain CC BY-NC-SA 4.0 attribution/terms. |
| Zhang et al., *Industrial Anomaly Detection with Domain Shift: A Real-world Dataset and Masked Multi-scale Reconstruction* (2023), [paper](https://arxiv.org/html/2304.02216v2), [release](https://github.com/zhangzilongc/MMR) | AeBAD data, changed capture conditions and a released domain-shift method. | Its paper already reports normal-region false positives under domain shift and small-defect misses; those observations alone are not our novelty. |

Optional mechanism context: Loaiza-Ganem et al., *Diagnosing and Fixing Manifold Overfitting in Deep Generative Models* (TMLR 2022), [paper](https://arxiv.org/abs/2204.07172). It provides established theory about degenerate likelihood behavior under dimensionality mismatch. Our experiments do not verify that this is the cause of the observed losses.

Our compact experiment differs in its 96-channel representation and same-sized middle-layer control. It tests one feature allocation choice within an established area of multi-scale anomaly detection; the result does not establish a broadly new method.
