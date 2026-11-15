# Weekly requirements and progress

Progress as of November 14, organized by the P16 course schedule. Some experiments finished later than planned. Methods and results are described in the [weekly project notes](PROJECT-WALKTHROUGH.md).

## The five experimental objectives

| Assignment objective | What we did | Result or limitation |
|---|---|---|
| FastFlow or CFLOW on all 15 MVTec categories | Chose FastFlow. Original ResNet and WideResNet results cover all categories and three repeats. [Complete tables](../final-evidence/team-tables/README.md). | Experiment complete; the paper’s headline scores are not all reproduced. The [code audit](notes/FASTFLOW-AUDIT.md) explains the comparison. |
| Compare ResNet-18, WideResNet-50 and EfficientNet-B4 | All three complete, plus a ResNet control matching one flow-filter setting. [Main chart](../final-evidence/team-plots/main-comparison.png). | Pretraining and training settings also differ, so the comparison does not isolate network size. |
| Propose and evaluate a multi-scale flow | One compact flow receiving 32 measurements from each of three feature levels, compared with 96 from the middle level. [Methods](../final-evidence/notes/METHODS.md). | Implemented and evaluated. No overall gain: 13.79 versus 14.57 pixel AP. This specific design is our project modification; multi-scale flows already exist. |
| Apply the method to a new domain | Fresh FastFlow fitted on normal AeBAD engine-blade pictures; three-repeat development comparison. [Domain tables](../final-evidence/transfer-team-tables/README.md). | Completed negative result. This is retraining on the new domain, not zero-shot transfer. Reserved images excluded. |
| Per-category overlays and failure analysis | Galleries cover every category; [bottle](../final-evidence/galleries/original-comparison/bottle_broken_large_000.jpg), [hazelnut](../final-evidence/galleries/compact/hazelnut_crack_000.jpg), [blade](../final-evidence/galleries/aebad/aebad-fracture-seed0.jpg) and [normal carpet](../final-evidence/galleries/original-comparison/carpet_good_000.jpg) selected as examples. | Figures and measured failures are available. Proposed causes and remedies remain untested. |

## The ten-week plan

| Week | Required in the P16 brief | Evidence and present status |
|---|---|---|
| 1–2 | Read papers; download MVTec; baseline repository setup; proposal due. | Paper review, data setup and implementation are complete. Literature notes are available. Confirm the proposal submission in the course portal. |
| 3 | Run baseline on all 15 categories; compute AUROC. | Earlier: original ResNet FastFlow, 45 runs. [All-category numbers](../final-evidence/team-tables/README.md). Completed. |
| 4 | Backbone ablation: ResNet-18 versus WideResNet-50. | Earlier: 45 WideResNet runs compared with ResNet. New full-study additions: EfficientNet-B4 and the matched ResNet control, 45 each. [Main chart](../final-evidence/team-plots/main-comparison.png). Completed. |
| 5 | Implement a multi-scale flow. | Earlier: compact three-level fusion and same-size middle-only control, 45 runs each. [Implementation explanation](../final-evidence/notes/METHODS.md). Completed. |
| 6 | Multi-scale results; anomaly-map visualizations; midterm due. | Earlier experiments remain the Week 6 evidence: [bottle comparison](../final-evidence/galleries/original-comparison/bottle_broken_large_000.jpg), [hazelnut comparison](../final-evidence/galleries/compact/hazelnut_crack_000.jpg), [category effects](../final-evidence/team-plots/category-changes.png). The completed results and figures also cover the Week 6 milestone. Confirm the midterm submission in the course portal. |
| 7 | Domain-transfer experiment. | Earlier small pilots, followed by the new complete three-repeat AeBAD application. [Engine-blade heatmaps](../final-evidence/galleries/aebad/aebad-fracture-seed0.jpg), [numbers](../final-evidence/transfer-team-tables/README.md). Complete, with poor localization and false alarms. |
| 8 | Failure analysis; per-category AUROC breakdown. | Tables include image and pixel AUROC and pixel AP per category. [Tables](../final-evidence/team-tables/README.md), [failure explanations](../final-evidence/notes/CLAIMS-AND-FAILURES.md). Category results and failure notes are complete; the final analysis still needs to be written. |
| 9 | Report; anomaly-localization heatmaps. | Results and figures are ready. The team still needs to write and review the report and prepare the presentation. |
| 10 | Final submission; final due. | Pending: final review, required declarations, repository link check, presentation and submission. Due November 27. |

## What changed in the recent completion phase

| Earlier completed main study | Added in the completion phase |
|---|---|
| Original ResNet: 45 runs | EfficientNet-B4: 45 runs |
| WideResNet: 45 runs | Matched ResNet control: 45 runs |
| Compact middle-only: 45 runs | PatchCore: 45 runs |
| Compact fusion: 45 runs | Full AeBAD application: 3 runs |
| 180 MVTec runs total | 135 additional MVTec runs + 3 AeBAD runs |

Combined total: **315 MVTec runs across seven configurations, plus three AeBAD runs**. Earlier AnomalyDINO and other exploratory pilots are additional work, not extra rows in that main comparison. The later code audit and combined tables added analysis without training further models.

[Project notes](PROJECT-WALKTHROUGH.md) · [Complete results](../final-evidence/README.md)
