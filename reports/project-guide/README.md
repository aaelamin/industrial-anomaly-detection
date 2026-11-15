# Project guide and progress

The [weekly project notes](PROJECT-WALKTHROUGH.md) cover the assignment, our model changes and the experiment results. The [requirements checklist](WEEK-BY-WEEK.md) tracks the remaining report and submission work.

| Document | Contents |
| --- | --- |
| [Weekly project notes](PROJECT-WALKTHROUGH.md) | Methods, model changes, results, diagrams and heatmaps |
| [Week-by-week requirements and status](WEEK-BY-WEEK.md) | The course’s ten-week plan, completed experiments and remaining report tasks |
| [Complete result tables](../final-evidence/team-tables/README.md) | Seven configurations, all 15 MVTec categories and three repeats |
| [Original comparison heatmaps](../final-evidence/galleries/original-comparison/HEATMAPS.md) | The same selected images shown for original ResNet, WideResNet and both compact models |
| [Compact comparison heatmaps](../final-evidence/galleries/compact/HEATMAPS.md) | Middle-only versus three-layer fusion |
| [Matched ResNet heatmaps](../final-evidence/matched-gallery/HEATMAPS.md) | The added recipe control and references |
| [Earlier blade reference comparison](notes/AEBAD-COMPARISON.md) | MMR, AnomalyDINO, PatchCore, masking failures and the missing FastFlow comparison |
| [Open questions](PROJECT-WALKTHROUGH.md#open-questions-and-remaining-work) | Unresolved explanations and proposed controls |
| [Engine-blade heatmaps](../final-evidence/galleries/aebad/HEATMAPS.md) | AeBAD normal and fracture examples |
| [Complete experiment evidence](../final-evidence/README.md) | All charts, galleries, methods, numbers and reproduction materials |

The main study includes 315 MVTec runs and three AeBAD FastFlow development runs. Earlier exploratory blade comparisons are reported separately. Compact fusion did not improve the average result, and the blade experiment showed substantial localization errors. The 1,479 reserved AeBAD images were not used.

The final report and presentation are in preparation. Submission is due November 27. Dataset credits and tool use are recorded in the [attribution notes](../final-evidence/notes/ATTRIBUTION.md).

Derived MVTec overlays: Bergmann et al., CC BY-NC-SA 4.0. AeBAD overlays: Zhang et al., CC BY 4.0. The earlier Motor diagnostic uses the M2AD release, whose dataset card declares Apache-2.0. The figures use the released defect annotations. Interpretation notes are in the [failure analysis](../final-evidence/notes/CLAIMS-AND-FAILURES.md).
