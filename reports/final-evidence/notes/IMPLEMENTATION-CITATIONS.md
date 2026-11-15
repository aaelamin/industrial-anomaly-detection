# Citations for the implementations and backbones

The BibTeX file includes the eight anomaly-detection/dataset references in the reading map and four implementation/backbone references. Cite the entries actually discussed in the report; the number of entries is not a reason to add unrelated claims.

- [ResNet, He et al.](https://arxiv.org/abs/1512.03385): residual feature extractor family. The supplied entry cites the inspected 2015 preprint.
- [Wide Residual Networks, Zagoruyko and Komodakis](https://arxiv.org/abs/1605.07146): wider residual networks. The supplied entry cites the inspected preprint. Our exact model is the recorded WideResNet-50-2 implementation and weight identity.
- [EfficientNet, Tan and Le, ICML 2019](https://proceedings.mlr.press/v97/tan19a.html): compound scaling and the backbone family. The paper's classification performance does not predict this study's anomaly-detection result.
- [Anomalib, Akcay et al.](https://arxiv.org/abs/2202.08341): implementation library. Its [official repository citation](https://github.com/open-edge-platform/anomalib/blob/main/README.md) identifies ICIP 2022, pages 1706-1710. This citation is additional to citing FastFlow and PatchCore.

Bibliographic metadata was checked against those primary sources on 13 November 2026 (Toronto). The CVF ResNet page was unavailable in this check, so the supplied entry uses the accessible author preprint. The installed source versions and weights in the reproduction package establish what actually ran; current library documentation does not replace that provenance.
