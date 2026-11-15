# Results and explanations

The reports cover the original weekly experiments and the completed comparison. Each weekly folder contains results and selected heatmaps; the final results include all seven configurations.

| Report | Contents |
| --- | --- |
| [Weekly project notes](project-guide/PROJECT-WALKTHROUGH.md) | Assignment requirements, model explanations, experiments and results |
| [Week-by-week requirements](project-guide/WEEK-BY-WEEK.md) | What was required, what is complete, and what remains |
| [Final experiment evidence](final-evidence/README.md) | Complete 315-run MVTec comparison and three AeBAD runs, with tables, galleries, methods and reproduction materials |
| [Week 3](week3/README.md) | Establish FastFlow with ResNet-18 on all 15 MVTec AD categories |
| [Week 4](week4/README.md) | Compare the WideResNet configuration and investigate weak categories |
| [Week 5](week5/README.md) | Implement feature fusion and its matched middle-level control |
| [Week 6](week6/README.md) | Present the completed multi-scale results, heatmaps and deliverable checks |
| [Compare the experiments](comparison/README.md) | The overall evidence, same-image heatmaps, limitations and midterm conclusions |

The original weekly study contains 45 Week 3 runs, 45 Week 4 runs and 90 Week 5 runs. The complete MVTec comparison adds 45 EfficientNet runs, 45 matched ResNet runs and 45 PatchCore reference fits, for 315 runs. Three AeBAD development runs are reported separately. All flow endpoints use epoch 500 and seeds 0, 1 and 2. Repetitions on the same images do not constitute independent datasets.

The CSV files contain the full category or run tables. `HEATMAPS.md` in each folder displays examples directly on GitHub. The comparison gallery shows the same images for all four configurations and all three seeds.

The raw maps, model weights and dataset remain outside Git because they are much larger. A fresh clone includes the results and figures needed to read this work, but not the saved models needed to replay it. See [running and changing experiments](../docs/RUNNING.md).
