# Faster batch pilot

This is the original setup or planning record. See the [completed results](README.md) for the later experiments and conclusions.

All 18 ten-epoch runs finished using a flow microbatch of 32. The effective batch remains 32. Frozen-backbone extraction and test inference still use four images at a time. The same saved PCA projections and paired initial model weights were used as in the original pilot.

| Category | Input | Microbatch 4 seconds/epoch | Microbatch 32 seconds/epoch |
| --- | --- | ---: | ---: |
| bottle | middle | 4.58 | 1.86 |
| bottle | fusion | 4.55 | 1.88 |
| screw | middle | 4.43 | 3.12 |
| screw | fusion | 4.31 | 3.12 |
| carpet | middle | 3.42 | 2.49 |
| carpet | fusion | 3.41 | 2.51 |

The median paired training speed ratio is 1.40 times the original pilot. These pilots ran at different times, so temperature and other system activity can affect that comparison.

The full 90-run, 500-epoch comparison is estimated at **27.6 training hours**, or **28.2 hours including modeled setup, checkpoints and evaluation**. The observed-rate total range is **27.7-28.9 hours**. Adding 25% scheduling headroom above the upper estimate gives **36.1 hours**. This is not a confidence interval or a guarantee.

The estimate adds the original measured PCA fitting cost because this validation reused saved projections. It excludes pauses, retries, prolonged thermal changes and dense map/preview exports.

Peak main-process allocated GPU memory was 2109 MiB. This excludes the separate recovery process and is not total GPU usage.

All runs passed finite-loss/gradient checks, full test-set finite-map checks, frozen-backbone checks and exact final-model reload checks. A fresh Python process reproduced the next full training epoch exactly from a checkpoint. Source hashes and all 18 output receipts were verified. Microbatch 32 need not reproduce microbatch 4 bit for bit because floating-point accumulation changes. Neither pilot establishes final accuracy or 500-epoch stability.

Microbatch 32 is suitable for the planned study on this GPU based on these short-run checks. The full-study controller, its retention/evaluation rules and a recovery smoke test remain to be prepared before launch. No full study has started.

The largest temporary category cache remains about 3.82 GiB, with about 1.66 GiB for planned model retention and active recovery checkpoints. Existing data, earlier studies, projections and exported maps are additional. Completed pilot feature caches are removed.

- [Measurements](fast/pilot.json)
- [Verification summary](fast/checks.json)
- [Research question and publication requirements](../../docs/WEEK5_PLAN.md#research-question-and-publication-direction)

Recheck or resume with `python scripts/week5_fast_pilot.py`. Rebuild this report with `python scripts/week5_fast_report.py`.
