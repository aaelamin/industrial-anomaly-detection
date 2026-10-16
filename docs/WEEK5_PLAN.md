# Week 5 plan

This is the original setup or planning record. See the [completed results](../reports/README.md) for the later experiments and conclusions.

Week 5 is the multi-scale flow implementation. Week 6 is the full multi-scale results and heatmap comparison for the midterm. The course guide, Project 16 on pages 22-23, specifically asks for a flow combining features from multiple CNN layers. Our plan should therefore include an actual feature-fusion experiment, not just a different average of the final heatmaps.

**Start by inspecting the levels in the saved models, then implement one small feature-fusion model with a matched single-level control. Benchmark it before committing to full training.**

## Why this follows from Week 4

The [follow-up](../reports/week4/followup.md) found clearer responses to some small defects, persistent background peaks on normal images, and a late loss spike in the weak pill run. We have not established that scale fusion caused these effects. The next question is whether giving one flow access to fine detail and broader context together helps it distinguish a defect from normal appearance variation.

The existing FastFlow CNN implementation already extracts three levels, applies a separate flow to each, and averages the resulting maps. At 256 x 256 input, the feature maps are 64 x 64, 32 x 32 and 16 x 16. Merely enabling all three would reproduce what we already do.

## First inspect the existing levels

Add a separate inference script that exposes the three maps before averaging. Verify that their unchanged average reproduces the saved prediction. Then evaluate the three individual levels, the three pairs and the existing three-level average. Keep maximum-pixel image scoring fixed.

Use all three seeds. Start with bottle, screw, hazelnut, carpet and pill so the inspection includes a strong category, persistent failures and regressions. Record individual map ranges and where the peaks fall. Run the saved models again to obtain the component maps because they cannot be recovered from the saved average.

This step requires inference, not training. It is an output ablation, not a newly trained single-level model. Report every subset. Do not use test AUROC to pick a different winning subset for each category. These test results have already guided our investigation, so the analysis is exploratory.

## The feature-fusion experiment

Use the frozen WideResNet-50-2 feature extractor and its existing reference weights for both arms. Combine features before the flow, rather than combining only its final outputs.

| Setting | Single-level control | Multi-scale variant |
| --- | --- | --- |
| Source features | Middle level only | Fine, middle and coarse levels |
| Channels after projection | 96 from the middle level | 32 from each level, concatenated to 96 |
| Common spatial size | 64 x 64 | 64 x 64 |
| Flow input | 96 x 64 x 64 | 96 x 64 x 64 |
| Downstream model | One normalization layer and one eight-block 2D flow | The same architecture |
| Output image score | Maximum of the resized anomaly map | The same rule |

Fit fixed PCA projections using normal training features only. For a reproducible first specification, sample up to 50,000 training feature vectors per level using a fixed sampling seed, save the sample identifiers and fitted projections, and use the top 96 components for the control or 32 per level for fusion. Standardize projected components using their training mean and standard deviation, with a fixed 1e-6 denominator floor. Record floor use. No test images or masks enter this fitting step.

Project channels at each level's native resolution, then resize the middle and coarse maps to 64 x 64 with bilinear interpolation. The fine map keeps its resolution. Concatenate along channels. Upsampling aligns the maps but does not create missing fine detail in the coarser features.

Use the same normalization, flow architecture, optimizer, epoch count, batch order and image preprocessing for the two arms. Start from identical downstream parameters for each paired seed. The initial protocol can use the Week 4 optimizer settings, with the microbatch chosen by memory checks and effective batch size held at 32. Freeze any pilot-driven training change for both arms before the full experiment.

The matched input size and channel count keep downstream capacity comparable. This tests the source and combination of features. It does not isolate every consequence of PCA and resampling, or prove that a difference from the original Week 4 model is caused only by fusion. Keep the existing FastFlow result as context and use the matched control for the main claim.

## What to finish during Week 5

1. Implement component-map extraction and the fixed subset comparison using the saved models.
2. Implement the two feature-input modes in a new model and training script. Keep the original study files and outputs unchanged.
3. Check projection provenance, shapes, finite outputs, backward gradients, checkpoint save/reload and exact resume. Verify that both arms have the same downstream parameter count and that the frozen backbone stays unchanged.
4. Run a short ten-epoch pilot for both arms on bottle, screw and carpet across all three seeds. Use it to check operation, loss behavior, memory, checkpoint size and speed. Ten epochs are not a final accuracy result. Do not choose settings by pilot test AUROC.
5. Estimate the full run from measured per-category timings and training counts, including feature extraction, checkpointing and evaluation. Agree on the compute budget before starting the full study.

The preliminary map inspection does not need training. The feature-fusion model does. A full comparison of two arms, 15 categories and three seeds would be 90 training runs at the chosen fixed epoch count. It may be substantially cheaper per run than the existing wide flows, but that needs measurement. Do not assume it will be quick or automatically start another multi-day job.

Week 6 is the full all-category comparison, mean and standard deviation across seeds, and matched heatmaps. If the budget is too large, revise the scope openly before running rather than quietly shortening only one arm or presenting a subset as the full study.

## Keep the conclusions fair

Keep image AUROC, pixel AUROC and pixel AP. Show all categories and seeds, including regressions. Use the same test masks, resizing and maximum-pixel image score. Add per-defect and descriptive defect-area breakdowns to check whether small defects benefit, without selecting model settings from those breakdowns.

The normal training images used for PCA are part of model fitting, not an independent validation set. MVTec test examples already informed this proposal. Any eventual paper claim needs further confirmation beyond repeatedly inspecting this same test set, potentially on BatterySurAD after checking the official split and prior FastFlow evaluations.

Retain a fixed set of late model checkpoints, for example epochs 400, 450, 475 and 500, and save per-level training diagnostics and gradient norms. Evaluate the prespecified final checkpoint for the reported comparison. The extra checkpoints are for investigating changes, not selecting the best test score.

## What stays separate

Pill seed 0's epoch-497 spike is a concrete reason to investigate training stability. A learning-rate schedule or gradient-clipping experiment should be a separately defined comparison, not an extra change made only to the fusion arm. The existing final models and results stay as recorded.

Normal-calibrated output averaging is another inexpensive candidate if individual map ranges differ substantially. It is a different experiment from combining features inside a flow. Do not mix it into the first feature-fusion comparison, and do not treat it as the entire course requirement.

CFLOW-AD, EfficientNet-B4 and the proposed domain-transfer work remain separate commitments. Week 5 does not require running every model at once.

## Reading and the paper direction

- [FastFlow](https://arxiv.org/abs/2111.07677) is the starting method. The pinned Anomalib 2.6.2 code determines the exact map construction used in our runs.
- [CFLOW-AD](https://arxiv.org/abs/2107.12571) uses conditional normalizing flows with multi-scale decoders. Its distinction from our baseline is not simply that it uses multiple feature levels. It is still a separate, unrun baseline.
- [MSFlow](https://arxiv.org/abs/2308.15300) is directly relevant. It uses parallel flows and a fusion flow to exchange information across scales, with different output aggregation for detection and localization. Its [official implementation](https://github.com/cool-xuan/msflow) is useful background for a larger architectural extension.

The small projected-feature concatenation experiment above is not MSFlow, and fusion by itself is not a novelty claim. It gives us a controlled course experiment and evidence for deciding what to investigate next. Keep the eventual paper improvement open rather than assuming this particular fusion design will be the contribution.

## Research question and publication direction

Our first question is: **At a fixed 96-channel input budget and identical downstream flow capacity, does joint modeling of fine, middle and coarse features improve anomaly detection and localization over middle-level features alone?** The proposed explanation is that local appearance and broader context provide complementary information about normality. This is a hypothesis, not an explanation already established by our heatmaps.

The existing FastFlow configuration estimates each feature level separately and averages its maps. Our experiment makes the levels available to the same flow before scoring. A matched middle-level control prevents an increase in downstream parameter count from explaining a difference. It does not establish superiority to every single level or isolate compression, interpolation and feature diversity from one another.

The design choices have different strengths of justification:

| Choice | Reason | Limitation |
| --- | --- | --- |
| Frozen backbone and shared projections | Keep representation fitting identical across paired seeds and avoid training the backbone on a small dataset | Conclusions depend on this pretrained representation |
| PCA fitted only on normal training features | Reduce channel cost with a reproducible, label-free transformation | High normal variance need not identify the dimensions most useful for detecting defects |
| 96 total channels, 32 per level | Match flow capacity and give each level an equal initial budget | Practical starting values, not an optimal allocation |
| Standardization before concatenation | Reduce differences in component scale | May amplify low-variance directions and does not make levels equally informative |
| Common 64 by 64 grid | Preserve the finest selected grid while aligning other levels | Upsampling adds no missing spatial detail and introduces correlated values |
| Same seeds, initialization and training schedule | Make differences easier to attribute to the feature input | Three seeds give limited information about variability |

PCA and normalization are preprocessing. The flow models the transformed representation. We do not claim an exact density over original images or an invertible end-to-end image transformation.

### What the full comparison should establish

Before starting, fix the configuration, categories, seeds, final epoch and reporting rules. Use mean category pixel AP as the primary localization comparison, with image AUROC and pixel AUROC reported alongside it. Report paired fusion-minus-control changes for every category and seed, overall means and seed variation. Pixels within an image are not independent experimental replicates. Three seeds should not be presented as strong statistical evidence by themselves.

Report regressions and training failures as well as improvements. Analyze defect size and background responses descriptively to see whether the proposed explanation fits the observations. Do not choose checkpoints or per-category settings by test performance. A positive average with contradictory category behavior calls for a narrower claim, not a claim that fusion always helps.

### What would support a paper

This comparison is a first experiment, not sufficient evidence of novelty. [MSFlow](https://arxiv.org/abs/2308.15300) already exchanges information across scales with parallel flows and a fusion flow. [CFLOW-AD](https://arxiv.org/abs/2107.12571) is a relevant conditional-flow baseline. [FastFlow](https://arxiv.org/abs/2111.07677) supplies the starting 2D flow approach. Our compact projected concatenation design is different from MSFlow, but being different does not establish a contribution.

A targeted literature search also located [UniFlow](https://www.mdpi.com/2078-2489/15/12/791), [MSAttnFlow](https://www.sciencedirect.com/science/article/pii/S0031320324009713) and [DualFlow](https://ieeexplore.ieee.org/abstract/document/10979555). Publisher search records describe multi-scale fusion or attention in these methods. Their full methods still need close reading before making a novelty claim. This search is not an exhaustive literature review.

If the first comparison is promising, the next evidence should include:

1. Fine-only and coarse-only controls at the same channel budget, plus a separately trained multi-level output-averaging control. These distinguish useful feature sources from joint modeling.
2. A small, prespecified compression study, including channel budget and projection choice, to test whether the result depends on our arbitrary 96-channel setting. Choose settings using a designated development protocol, not repeated MVTec test inspection.
3. Relevant published baselines under clearly documented settings. A gain over our weaker reproduced baseline alone is not enough. Measure end-to-end inference, peak memory and preparation cost on the same hardware if efficiency is part of the claim. Cached flow-training speed is not deployment speed.
4. Confirmation on a second dataset whose outcomes have not guided the design, with its split and training regime defined in advance. Distinguish retraining on that dataset from transfer without retraining.

A possible paper direction is the accuracy and compute tradeoff of compact joint feature modeling, or an explanation of when scale fusion helps and fails. Neither is an established gap yet. If these results are weak, they still satisfy a useful course experiment and can guide a better improvement rather than forcing a publication story around this architecture.

## Running the implementation

The model and projection code are in `scripts/week5_model.py`, with settings in `configs/week5.json`. The PCA implementation uses randomized SVD with four power iterations and a fixed sampling seed. Component means and standard deviations come from the same sampled normal training vectors. The middle-level PCA is fitted once to 96 components, with its first 32 shared by the fusion arm. Projections are shared across seeds.

From the repository folder, using the project environment:

```sh
python -m unittest discover -s tests -p test_week5.py -v
python scripts/week5_pilot.py
python scripts/week5_levels.py
python scripts/week5_report.py
```

The pilot is limited to the agreed 18 ten-epoch runs. It verifies completed outputs and resumes incomplete runs from an epoch checkpoint. It checks full test-set inference for timing and finite outputs, but does not use short-run test accuracy to select settings. The feature-level script evaluates the 15 saved Week 4 models for the five inspection categories. It does not train.

The faster setting has a separate configuration and pilot so the original measurements remain reproducible:

```sh
python scripts/week5_fast_pilot.py
python scripts/week5_fast_report.py
```

This uses `configs/week5_fast.json`, a flow microbatch of 32 and an unchanged effective batch of 32. Feature extraction and test inference keep image batches of four. It reuses the saved training-only projections, repeats all 18 ten-epoch runs and the fresh-process recovery check, and writes to `results/week5/fast-pilot/` and `reports/week5/fast/`. The runner is a frozen companion to the original pilot with separate paths and image-batch handling. Both call the same model and training implementation. It cannot launch the full study.

Pilot outputs are kept under `results/week5/pilot/`, with small summaries under `reports/week5/`. Temporary feature files are under `work/week5-pilot/` and are removed after a category finishes. Dataset files and the earlier studies remain unchanged.

## Full comparison

`scripts/week5_study.py` uses `configs/week5_study.json`: both inputs, all 15 categories, seeds 0, 1 and 2, 500 epochs, effective batch 32 and flow microbatch 32. It starts each model from scratch. The two-epoch controller check uses separate `study-smoke` folders and is excluded from the results.

```sh
python scripts/week5_study.py
```

The same command resumes saved work and verifies completed runs. It refuses changes to the saved configuration or source files. Only one controller can write to the study at a time.

Progress is in `work/week5-study/status.json`. Results are in `results/week5/study/<category>/seed_<n>/<arm>/`. Each run saves its loss and gradient history, final metrics, image scores, compressed anomaly maps and selected heatmaps. Models at epochs 400, 450, 475 and 500 are retained. Only epoch 500 is used for the reported evaluation. The earlier retained models are for later diagnosis.

Recovery checkpoints keep the two latest generations, saved every 25 epochs and at a requested pause. An unexpected interruption can lose up to 24 completed epochs. To request a pause, create the empty file `work/week5-study/PAUSE`. The runner saves at the next epoch boundary and stops. Remove that file before resuming with the command above.

Temporary features are shared across the six runs in a category and removed when all six finish. Completed runs keep their model files but remove optimizer recovery files. An interrupted category keeps its verified feature cache for resume. Existing datasets and earlier study outputs are not deleted. Dense maps and heatmaps require additional storage beyond the model-only pilot estimate. The runner checks for at least 8 GiB free before preparing each category.

The process prevents idle sleep while running. Shutdown, forced sleep or a runtime error can still stop it. Check the status before assuming it is running after one of those events.
