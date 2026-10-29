# Week 6 verification notes

[Week 6 report](README.md) | [Verification record](verification.json)

Checked against the saved evidence on October 27, 2026. The [machine-readable verification](verification.json) records the scope.

| Item | Finding |
| --- | --- |
| Full multi-scale comparison | Complete unique grid of 90 runs, 15 categories, seeds 0/1/2 and both arms. All completion receipts use the recorded protocol. |
| Reported averages and variability | Recalculated from the per-run table; all means and sample standard deviations match the saved summary within 1e-12. Every run's table values match its metrics file. |
| Saved run evidence | SHA256 checks passed for 270 metrics, score and training-history files. Final checkpoint and saved-map files exist for all 90 runs. Large model/map contents were not rehashed or re-evaluated in this audit. |
| Visualization coverage | Both existing galleries contain 36 readable images and normal/defective examples for all 15 categories. Bottle and hazelnut sheets were also visually reviewed during this check. |
| Interpretation | The negative overall result, category differences, three-seed variability and limitations are documented. The original endpoint is preserved. |
| Reproduction instructions | Environment, settings, seeds and commands are documented. The full Week 5 controller requires Windows/CUDA and local data; this check did not reproduce the study on a clean machine. |
| Midterm content | The local update source contains methods, results, literature, challenges, a revised plan and a contribution declaration. Presence of a contribution table does not verify the accuracy of individual contributions. |
| Standalone midterm package | The local **update ZIP** is readable and matches its local source, but contains only `main.tex` and the new comparison figure. It lacks referenced `figures/screw.jpg` and `figures/hazelnut.jpg`, and contains no PDF. Compilation of that local update folder fails at the missing screw figure. |
| Actual course submission | Not verified by this repository check. The update ZIP may have been intended to overlay an existing complete project; its missing files do not establish that the submitted midterm was incomplete. |

The original full analysis separately records metric recomputation, map/receipt verification, 15 projection checks and 45 paired initialization checks. Those earlier checks are distinguished from the lighter current audit in the verification record. No training, new model inference, reserved-image access or midterm-source editing was performed for this update.


The original local midterm update package needs its two referenced figures and a compiled PDF to serve as a standalone deliverable. It may be an overlay for an existing project; the actual submitted package has not been verified. This documentation update does not alter the midterm source.
