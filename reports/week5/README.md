# Week 5: combine feature levels before the flow

[All weeks](../README.md) | [Full analysis and design rationale](week5.md) | [Heatmaps](HEATMAPS.md)

## What we needed to do

Implement a multi-scale flow experiment. Our schedule puts implementation in Week 5 and its results and visualizations in Week 6. The full comparison is now available for the midterm.

FastFlow already used three feature levels. Each level had its own flow and the output maps were averaged. We tested a different question: would giving one flow the fine, middle and coarse features together improve its predictions?

## How we made that comparison

Both versions used the same frozen WideResNet backbone as Week 4. A fixed PCA projection compressed the features using normal training data only.

| Model | Features going into the flow |
| --- | --- |
| Middle-only control | 96 compressed components from the middle level |
| Fusion | 32 components from each of three levels, giving 96 total |

Both inputs were aligned to a 64 by 64 grid, then passed through the same normalization layer and eight-block flow. Each had 1,065,600 trainable parameters. Paired runs used identical starting parameters and training order. We ran both versions on all 15 categories with three seeds and 500 epochs, for 90 runs.

Keeping the channel count and downstream model fixed makes this a useful comparison of feature inputs. It does not make either new model equivalent to the original three-flow FastFlow model.

## Why this design, and how strong is that justification?

The three levels were inherited from the existing FastFlow extractor, rather than chosen by searching test scores. They span fine spatial detail and broader context. The middle level provided a simple control between those extremes. It was not shown to be the best single level.

Compression made a full paired experiment practical on the available GPU. Equal allocation gave every level a defined starting budget. Those are sensible engineering reasons to try the design, but they are not evidence that 96 channels, 32 per level, or PCA are optimal. We did not compare all alternatives before choosing them.

The [full rationale](week5.md#why-these-layers-and-this-structure) explains the actual layers, alternatives, limitations and research references. The important distinction is that we tested a defensible hypothesis, not an architecture already known to improve the baseline.

## What we got

| Metric | Middle-only control | Fusion | Change |
| --- | ---: | ---: | ---: |
| Image AUROC | 71.76 ± 3.39% | 67.55 ± 1.66% | -4.21 points |
| Pixel AUROC | 73.89 ± 1.41% | 69.80 ± 1.68% | -4.09 points |
| Pixel AP | 14.57 ± 1.68% | 13.79 ± 1.04% | -0.78 points |

Fusion did not improve the overall result. Six categories improved in pixel AP, but nine declined. All five texture categories declined in mean pixel AP. Both new models were far weaker than Week 4, so the shared changes to the representation and flow structure need attention too.

The [full analysis](week5.md) includes every category, seed differences, actual heatmaps, defect-size breakdowns and the limits of the conclusion. The [cross-week comparison](../comparison/README.md) shows how the same images changed across all configurations.

## Where that leaves us

Keep Week 4 as the stronger reference. Present Week 5 as a completed controlled experiment with a negative overall outcome. Do not claim that all multi-scale fusion fails, or that this design is a publishable improvement.

The most useful next step is to separate representation choices from training behavior. The [retained-checkpoint analysis](checkpoints/README.md) checked late behavior without rerunning training. Hazelnut and screw were already weak at epoch 400, while carpet fluctuated substantially. After that, a small experiment could test one change such as the projection size or spatial alignment. A different fusion design is also open for discussion, but it needs its own comparison and should not overwrite these results.

The compact models are much cheaper to train than Week 4. A full 90-run study is still substantial, with the original pilot estimating about 28 hours plus overhead and pauses. A targeted experiment on a few categories would be a better first step than another full sweep.

## Results and implementation

- [Full analysis, decisions and conclusions](week5.md)
- [Category table](analysis/per_category.csv), [all 90 runs](analysis/per_run.csv) and [defect table](analysis/per_defect.csv)
- [Heatmaps for all 15 categories and both variants](HEATMAPS.md)
- [Existing feature-level inspection](levels.md)
- [Original pilot](pilot.md) and [faster-batch pilot](fast_pilot.md), both historical setup records
- [Model code](../../scripts/week5_model.py), [study settings](../../configs/week5_study.json), [original experiment plan](../../docs/WEEK5_PLAN.md)
- [How to run or propose a changed experiment](../../docs/RUNNING.md)

Full local outputs, projections and retained checkpoints are under `results/week5/study/`. The GitHub heatmaps are selected examples. They can be viewed without downloading model weights or installing Python.
