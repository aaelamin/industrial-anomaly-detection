# Results and explanations

Read these in order. Each weekly folder explains the task, the model, the choices, the results and what followed from them. No installation is needed to read the reports or view the included heatmaps on GitHub.

| Read | What it covers |
| --- | --- |
| [Week 3](week3/README.md) | Establish FastFlow with ResNet-18 on all 15 MVTec AD categories |
| [Week 4](week4/README.md) | Compare the WideResNet configuration and investigate weak categories |
| [Week 5](week5/README.md) | Test feature fusion against a matched middle-level control |
| [Compare the experiments](comparison/README.md) | The overall evidence, same-image heatmaps, limitations and midterm conclusions |

There are 45 Week 3 runs, 45 Week 4 runs and 90 Week 5 runs. All use seeds 0, 1 and 2 and the final model after 500 epochs. These are repetitions on the same dataset, not 180 independent datasets.

The CSV files contain the full category or run tables. `HEATMAPS.md` in each folder displays examples directly on GitHub. The comparison gallery shows the same images for all four configurations and all three seeds.

The raw maps, model weights and dataset remain outside Git because they are much larger. A fresh clone includes the results and figures needed to read this work, but not the saved models needed to replay it. See [running and changing experiments](../docs/RUNNING.md).
