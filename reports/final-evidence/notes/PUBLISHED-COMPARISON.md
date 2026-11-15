# Published numbers and reproduction scope

The [comparison table](../team-tables/published-context-comparison.csv) lists the published configurations beside our results. Published values are point estimates; our entries use three-seed means and sample SD. Pixel AP is left blank where the papers do not report it.

FastFlow Table 5 reports ResNet-18 all-3x3 image/pixel AUROC of 97.9/97.2%, alternating ResNet 97.3/96.8%, and alternating WideResNet 99.3/98.1%. Its 99.4% headline uses CaiT. **The kernel-pattern comparison already exists in the original paper**; our matched control addresses comparability within the class study and is not a novel contribution. [FastFlow, Tables 5–6](https://arxiv.org/html/2111.07677).

The PatchCore 1% default reports 99.0% image and 98.0% pixel AUROC. Its higher headline involves other settings. The paper's default image processing and our common 256-pixel grid differ. [PatchCore, Tables 1–2](https://openaccess.thecvf.com/content/CVPR2022/papers/Roth_Towards_Total_Recall_in_Industrial_Anomaly_Detection_CVPR_2022_paper.pdf).

Our adaptations cover all 15 categories, but do not reproduce the published headline scores. Known configuration differences are documented; their individual effects have not all been isolated.
