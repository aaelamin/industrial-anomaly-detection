# Week 5 pilot

This is the original setup or planning record. See the [completed results](README.md) for the later experiments and conclusions.

This is the original microbatch-4 pilot. The later [microbatch-32 validation](fast_pilot.md) passed all 18 runs and provides the revised time estimate.

Both feature inputs use the same 96-channel flow architecture. The middle-only control uses 96 projected middle-level channels. Fusion uses 32 projected channels from each of three levels. PCA and standardization were fitted on sampled normal training features only.

Each arm has 1,065,600 trainable parameters. The pretrained feature extractor and fitted PCA projections stay frozen.

The pilot ran both arms for ten epochs on bottle, screw and carpet with seeds 0, 1 and 2. These 18 runs check operation and cost. Their short-run accuracy was not used to choose settings. No full training study has been started.

## Measured training time

The table uses the median of epochs 2-10 for each run, then the median across its three seeds. Preparation, evaluation and checkpoint writes are timed separately.

| Category | Training images | Steps per epoch | Middle seconds per epoch | Fusion seconds per epoch |
| --- | ---: | ---: | ---: | ---: |
| bottle | 209 | 6 | 4.58 | 4.55 |
| screw | 320 | 10 | 4.43 | 4.31 |
| carpet | 280 | 8 | 3.42 | 3.41 |

## Estimate for the full comparison

At 500 epochs, two arms, 15 categories and three seeds, the full comparison would contain 90 runs. Scaling the measured time per optimizer step by the actual training counts gives **38.9 hours of training**, with **35.8-68.8 hours** from the fastest and slowest observed pilot rates.

Shared feature extraction and projection fitting: about 0.30 hours. Checkpoint writes every 25 epochs: about 0.05 hours. Model inference on the test images: about 0.16 hours.

The feature-level inspection also measured metric computation. Extrapolating its time per image gives roughly 0.10 additional hours for image AUROC, pixel AUROC and pixel AP across the full comparison. This uses different predicted maps and test group sizes, so it is approximate.

The modeled total is **39.5 hours**, with an observed-rate range of **36.4-69.4 hours**. Allowing 25% planning headroom above the upper estimate gives **86.8 hours**. That allowance is a scheduling choice, not a measured confidence bound.

The estimate assumes category feature caches are shared across seeds and arms. It excludes pauses, retries, long-run thermal changes, preview/map exports and most process startup. A ten-epoch pilot cannot establish 500-epoch stability or final accuracy.

## Memory and storage

The largest measured allocated GPU memory was 613 MiB. This is PyTorch allocation in the main process, not total system or GPU usage, and excludes the separate recovery-check process.

The largest category's native and projected training caches are estimated at 3.82 GiB. They are temporary and should be processed one category at a time. Keeping four late/final models per run plus two active recovery checkpoints is estimated at 1.66 GiB. Dataset storage, existing studies, projections and exported test maps are additional.

The completed pilot retains projections, final models and the latest two recovery checkpoints for each run. Its completed category feature caches were removed. Old studies and their data were left unchanged.

## Checks

All 4 targeted tests passed. They check projection algebra and feature routing, rejection of test images during fitting, constant-feature handling, gradient accumulation, and continuation with reconstructed training objects.

A separate Python process loaded the bottle control checkpoint and reproduced the next full epoch exactly, including model weights, optimizer state, random states and training diagnostics. All 18 final models also reproduced their saved-model probe predictions exactly after reload. The frozen backbone remained unchanged, and each seed pair had identical initial downstream weights and trainable parameter counts.

PCA sample identifiers, training-image hashes and scale-floor counts are saved with each projection. Disk-cached projected features matched a fresh extraction exactly. Full pilot test-set inference produced finite maps with the expected shape. This checks operation, not final quality.

## Next step

Review the measured budget before starting a full comparison. The inference-only [feature-level results](levels.md) help interpret the existing models, but should not be used to silently change the pilot design or select a different best subset for each category. The feature-fusion proposal remains an experiment rather than an established improvement.

- [Per-run timing and memory](pilot_runs.csv)
- [Detailed benchmark and preparation times](pilot.json)
- [Implementation and experiment plan](../../docs/WEEK5_PLAN.md)

Run `python scripts/week5_pilot.py` to check or resume this fixed ten-epoch pilot. Finished runs are checksum-verified and skipped. Run `python scripts/week5_levels.py` for the saved-model level inspection, then `python scripts/week5_report.py` to rebuild this report. There is no full-study launch option in the pilot command.

## Extra batch-size timing check

We also timed microbatches of 4, 8, 16 and 32 while keeping the effective batch at 32. Each setting repeated 32 normal bottle images for 12 optimizer steps, excluding the first three from timing. This is a throughput check, not another accuracy experiment.

| Input | Microbatch | Seconds per step | Peak allocated MiB |
| --- | ---: | ---: | ---: |
| middle | 4 | 0.412 | 609 |
| middle | 8 | 0.338 | 809 |
| middle | 16 | 0.312 | 1211 |
| middle | 32 | 0.305 | 2109 |
| fusion | 4 | 0.408 | 609 |
| fusion | 8 | 0.350 | 809 |
| fusion | 16 | 0.317 | 1211 |
| fusion | 32 | 0.313 | 2109 |

The main estimate above remains based on the 18-run microbatch-4 pilot. A faster setting here would still need its own longer pilot and recovery check before changing the study configuration.

- [Batch-size timing measurements](batch_probe.json)
