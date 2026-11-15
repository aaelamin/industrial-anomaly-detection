# Main findings

These results average all 15 categories within each seed, then summarize the three seed averages. We did not conduct a statistical significance test.

| Question | Result | Interpretation |
|---|---|---|
| Does the 96-channel fusion design improve on its matched middle-layer control? | Pixel AP 13.79 ± 1.04% versus 14.57 ± 1.68%; paired change −0.78 ± 2.23 points. Image AUROC and pixel AUROC also decline on the macro comparison. | This compact design shows no aggregate benefit. Individual categories have both gains and losses; the result does not cover other multi-scale methods. |
| Does matching the ResNet coupling-kernel pattern close the reference gap? | Matched ResNet pixel AP 39.11 ± 0.54% versus original ResNet 39.69 ± 1.04%; paired change −0.59 ± 1.46 points. | This control does not close the gap. The observed difference is small and varies by seed/category. It does not isolate all implementation details. |
| Does the WideResNet advantage remain in the matched comparison? | WideResNet pixel AP 52.99 ± 2.95%, 13.88 ± 3.49 points above matched ResNet. All three seed macro differences are positive. | The recorded configuration advantage remains. Different pretrained weights and execution details prevent a backbone-only causal claim. |
| Is there a strong second published-method reference? | PatchCore: image AUROC 98.40 ± 0.05%, pixel AUROC 97.88 ± 0.02%, pixel AP 56.13 ± 0.06%. Its pixel AP exceeds WideResNet by 3.14 ± 2.90 points. | PatchCore is the strongest aggregate reference here. It is not best on every category, and its seed variation comes from coreset sampling. |
| Does refitting on AeBAD establish a robust new-domain detector? | Final pixel AP 2.30 ± 0.50%, PRO at 5% FPR 14.66 ± 0.41%, normal-pixel FPR 13.62 ± 2.56%. Previously detected small regions are lost in every seed, despite net detection gains. | No. Region gains accompany losses and increased false alarms. Initialization is a diagnostic control, not a competitive baseline. |

All ± values above are sample SD across three seeds. Paired-change SD is computed from the three seed differences, not from separately adding the endpoint SDs. Reported percentages and percentage-point differences have different meanings. The full CSVs retain unrounded numbers.

The published FastFlow image-level scores are not reproduced by these adaptations. The completed coupling-pattern control narrows one possible explanation, but the remaining paper-to-implementation gap is unresolved.

The planned comparison, backbone study, multi-scale ablation, AeBAD application and failure analysis are complete. The report and presentation remain to be written. The [related-work review](LITERATURE.md) places the tested design alongside existing methods.
