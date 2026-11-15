# Final experiment results

We completed 315 MVTec runs across seven configurations and three AeBAD development runs. Every flow result uses the planned 500-epoch endpoint.

## Results and figures

- [Main findings](notes/FINAL-FINDINGS.md), [main and per-category tables](team-tables/README.md), [all 315 runs](team-tables/all-315-runs.csv), and [LaTeX table](team-tables/main-table.tex).
- [Main comparison](team-plots/main-comparison.png), [category differences](team-plots/category-changes.png), and [figure captions](team-plots/CAPTIONS.md). Editable SVGs and [plotting instructions](analysis/README.md) are also included.
- [Original comparison heatmaps](galleries/original-comparison/HEATMAPS.md), [compact-model heatmaps](galleries/compact/HEATMAPS.md), [matched ResNet heatmaps](matched-gallery/HEATMAPS.md), and [AeBAD examples](galleries/aebad/HEATMAPS.md). Each gallery also has an HTML index for local viewing.
- [AeBAD results and region changes](transfer-team-tables/README.md), with [numeric checks](transfer-evidence/README.md).

## Methods and references

- [Methods and equations](notes/METHODS.md), [implementation settings](implementation-recipes.json), and [failure analysis](notes/CLAIMS-AND-FAILURES.md).
- [Related work](notes/LITERATURE.md), [BibTeX references](notes/references.bib), and [attribution and tool use](notes/ATTRIBUTION.md).
- [Reproduction instructions](reproduction/README.md), [environment setup](reproduction/SETUP.md), [data and weights](notes/DATA-AND-WEIGHTS.md), [test status](reproduction/QUALIFICATION-STATUS.json), and [restart checks](verification/fresh-output-bootstrap/README.md).

## Checking the saved results

Run `python reports/final-evidence/verify_all.py` from the repository root. Only the Python standard library is needed. The script checks file hashes, local links and the reported numbers without loading models or images. Training and inference are separate steps described in the reproduction instructions.

The restart checks passed on the original machine. A clean installation on another machine and a second execution of all 318 runs have not been completed. The test records distinguish those tasks from checking the saved tables.

## Conclusions and remaining work

The compact fusion model did not improve the average result over its matched control. The AeBAD experiment showed localization errors and false alarms, with both small-defect gains and losses. Neither experiment established an improved detector. Heatmap colors are scaled separately in each panel and are not defect probabilities. The 1,479 reserved AeBAD images were not used.

Final submission is due November 27. The report and presentation are still being prepared.

MVTec images and overlays: Bergmann et al., CC BY-NC-SA 4.0. AeBAD overlays: Zhang et al., CC BY 4.0. Model weights, full datasets, personal configuration files and reserved-image content are not included.
