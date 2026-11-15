# Reproduce the reported numeric aggregation

This bundle contains saved per-run metrics for 7 complete configurations: 15 MVTec AD categories and three seeds each (315 runs). It contains no source images, masks, model weights, checkpoints, personal paths, or reserved AeBAD inputs.

From any directory, using Python 3.10 or later:

```text
python /path/to/evidence-bundle/verify.py --output-dir /path/to/new-results
```

On Windows, quote paths containing spaces. Only Python's standard library is needed. The output directory is optional; without it, the script verifies the evidence and prints a brief result. Generated files are `verified-results.json` and `verified-results.md`.

The verifier checks file hashes, unique model/category/seed identities, complete coverage, finite metric ranges, macro averages, sample standard deviations and the paired fusion-minus-control changes. A full-model average first averages equally across the 15 categories within each seed, then reports the mean and sample SD of those three seed averages. It does not pool all test pixels across categories or mistake category variation for seed uncertainty.

Model identifiers:

| Identifier | Configuration |
|---|---|
| `week3` | Original ResNet-18 FastFlow configuration |
| `week4` | Original WideResNet-50-2 FastFlow configuration |
| `middle` | Compact middle-layer control |
| `fusion` | Compact multi-layer fusion |
| `patchcore` | Anomalib 2.6.2 PatchCore adaptation, 1% coreset, WRN layer2/3, 256 input and metric grid |
| `efficientnet` | EfficientNet-B4 FastFlow, alternating coupling kernels, pretrained ra2 weights adapted to the common 256 input |
| `resnet` | ResNet-18 FastFlow follow-up with alternating coupling kernels, original ResNet weights |

The original ResNet and WideResNet configurations differ in pretrained weights and coupling-kernel settings, so their contrast is not an isolated backbone effect. PatchCore seeds affect coreset construction; they are not flow-training seeds. The PatchCore adaptation is not a claim to reproduce the authors' headline result. The compact fusion comparison uses its matched middle-layer control. Its negative mean AP change is retained.

`original-per_run.csv`, `original-summary.json` and `original-paired_changes.csv` are preserved copies of the completed class comparison. `additional-per-run.csv` exports only fully completed additional model groups; incomplete model groups are excluded. `completed-source-identities.json` retains the completion-receipt hash for every exported additional run. `manifest.json` records the expected source hashes and numeric summaries.

This is an audit of aggregation from saved metrics. It does **not** reproduce raw-image inference, training, mask resizing, pixel-metric computation, or the file-access audit. Hash agreement demonstrates consistency with this bundle's recorded inputs, not an independently certified chain of custody. Reproducing the models still requires the documented environments, licensed datasets, weights and training configuration. Fresh-machine training portability remains unverified. Reported SD is algorithmic seed variability, not a population confidence interval. No statistical significance or new-method claim is inferred here.

[Attribution notes](../notes/ATTRIBUTION.md) record dataset credits and tool use.
