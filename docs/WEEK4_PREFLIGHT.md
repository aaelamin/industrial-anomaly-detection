# Week 4 setup and bottle benchmark

This is the original setup or planning record. See the [completed results](../reports/README.md) for the later experiments and conclusions.

The next comparison is ResNet-18 against WideResNet-50. This preflight checks whether the larger model fits on the laptop and estimates the cost before starting the full study. It does not produce a trained baseline result.

## Repository check

Fetched `origin` on 2026-09-29. Remote `main` is still at `b920deb`, the Week 3 commit. There were no newer commits on `main` to incorporate. Unpushed work on teammates' computers is not visible in this check.

## Settings

The proposed configuration is in `configs/week4.json`.

| Setting | WideResNet experiment |
| --- | --- |
| Backbone | `wide_resnet50_2` |
| Pretrained weights | `wide_resnet50_racm-8234f177.pth`, the timm 0.5.4 default |
| Input | 256 x 256, same resize and normalization as Week 3 |
| Feature layers | layer1, layer2, layer3 |
| Flow steps | 8 per feature level |
| Flow convolutions | Alternating 3 x 3 and 1 x 1 |
| Hidden ratio | 1.0 |
| Training plan | 500 epochs; seeds 0, 1, 2; normal training images only |
| Optimizer | Adam, learning rate 0.001, weight decay 0.00001 |
| Batch | Effective 32, initially testing microbatches of 4 |
| Evaluation | Final checkpoint, maximum-map image score, same pixel metrics |

Sources: [WideResNet reference configuration](https://github.com/gathierry/FastFlow/blob/master/configs/wide_resnet50_2.yaml) and [timm 0.5.4 weights](https://github.com/huggingface/pytorch-image-models/blob/v0.5.4/timm/models/resnet.py). The downloaded weights were checked against the published checksum prefix; the full checksum is recorded in our config.

The reference uses all 3 x 3 flow layers for ResNet-18 and alternating kernels for WideResNet. Comparing these settings tests the two reference configurations, not the backbone alone. The pretrained weights also come from different training recipes. We should describe these differences in the comparison; isolating only the architecture would need a separate control with matched flow settings.

## Benchmark

```sh
python scripts/week4_preflight.py --micro-batch-size 4 --epochs 3
```

This trains on the bottle training split for three epochs and evaluates all 83 bottle test images. It checks that a sampled batch of cached features matches direct extraction and that saving and reloading the checkpoint preserves predictions. The cache is stored as float32 arrays on disk to limit RAM use. The effective batch remains 32.

Outputs are kept under `work/week4-preflight/`, outside the baseline results. `benchmark.json` records training times, peak GPU memory, cache size, checkpoint size and a runtime estimate. The first epoch is excluded from the steady-state estimate. The extrapolation uses each category's number of full training batches and does not represent a measured Colab runtime.

The short-run AUROC values are only an evaluation check. They should not be compared with the 500-epoch Week 3 results or used to choose settings.

## Choosing a machine

Colab hardware and usage limits vary. A free session can run for at most 12 hours and may end earlier. Checkpoints need persistent storage so a disconnected session does not lose the run. A Colab GPU is not automatically faster than the laptop; run the same preflight on the assigned GPU before comparing times. See [Colab's resource limits](https://research.google.com/colaboratory/faq.html#resource-limits).

The full study has not been started.

## Laptop measurements

The RTX 4050 Laptop GPU passed the bottle preflight on 2026-09-29. Three complete training epochs, checkpoint save/reload, and evaluation of all 83 test images succeeded. Saved predictions were unchanged after reload. Sampled disk-cached features matched direct extraction exactly.

| Measurement | Result |
| --- | ---: |
| Warmed-up bottle epoch, microbatch 4 | About 13-16 seconds |
| Bottle, 500 epochs, one seed | About 2 hours, estimated |
| All 45 runs | About 100-110 GPU hours, estimated, plus setup and evaluation |
| Peak allocated GPU memory, microbatch 4 | 2,009 MiB |
| Peak allocated GPU memory, microbatch 8 | 3,104 MiB |
| Bottle feature cache | 1.43 GiB |
| Largest category feature cache | About 2.67 GiB |
| One checkpoint including optimizer | 694 MiB |
| 45 retained checkpoints alone | About 30.5 GiB |

A second test compared three epochs each with microbatches of four and eight. Their warmed-up medians were 14.46 and 14.33 seconds. This small difference is not enough to justify the higher memory use, so the proposed config keeps four. Both use effective batches of 32.

These are short benchmarks. Longer training, heat, other programs, and interruptions may change the timings. The preflight AUROC values are not baseline results.

### Local or Colab?

The GPU can run this model. The full study would occupy it for roughly four to five days of uninterrupted training, and storage needs attention first. C: had only 3.9 GiB free before the benchmark. Keeping checkpoints, separate model files, maps and one reusable feature cache would need roughly 50-60 GiB of additional free space with the current saving approach. Even keeping only the 45 checkpoints exceeds the available space.

Do not start the full study locally with the current free space. Local is still an option after freeing or adding storage, if leaving the laptop running for several days is acceptable. For Colab, first run the same preflight on the GPU actually assigned. We have not measured Colab speed, and a free GPU is not guaranteed to be faster. Its session limits mean the full study will need persistent checkpoints and multiple sessions.

The large temporary feature cache and checkpoint from this preflight were removed after verification to recover space. Small outputs remain in `work/week4-preflight/`; measurements are also saved in [preflight.json](../reports/week4/preflight.json). No Week 3 artifacts were removed or changed. The full Week 4 runner and Colab persistence setup still need to be prepared after choosing where to train. No full study has started, and these new files have not been pushed.
