# Failure analysis and limitations

The compact model and the AeBAD application both produced substantial errors. Their interpretation is limited by the model changes, repeated use of the development data and uncertainty about physical-object independence.

| Evidence | Supported statement | Unsupported extension |
|---|---|---|
| Matched compact middle/fusion comparison | This 96-channel fusion design did not establish an aggregate pixel-AP benefit over its matched middle-layer control. | Multi-scale flows do not work; fusion causes the entire gap to the original WideResNet reference. |
| Complete backbone/configuration table | These recorded adaptations differ on the common input and metric grid. | Backbone size alone caused the difference; our configuration exactly reproduces the published FastFlow setup. |
| AeBAD endpoint analysis | Ranking worsened and normal-pixel false alarms increased; some previously detected regions were lost despite net detection gains. | Every small defect got worse; fracture recall fell for every seed; nondefect pixels are all physical background. |
| Normal-only threshold calibration | Evaluation defect labels did not select the frozen operating point. | The same development sample is independent confirmation, or 99th-percentile calibration guarantees a 1% false-positive rate under changed capture conditions. |
| Recovery and integrity receipts | The named tested routes preserve state/predictions through the tested interruptions, and saved tables are internally consistent. | A clean machine or every full run has been independently reproduced. |
| Literature and rejected pilots | Several plausible ideas failed their development controls; the failures remain documented. | A novel detector, mechanism, or publishable improvement has been established. |

## Three failure examples

1. **Responses on an intact texture.** The normal carpet gallery has an empty released defect mask and spatially varying responses. This is a visual observation about that image, not a calibrated false-alarm count. The complete compact-flow texture category results provide the broader comparison. A larger retained feature budget is a possible follow-up, but it has not been validated as a remedy.
2. **Poor localization on a cracked object.** The filename-selected hazelnut crack sheet shows compact predictions away from part of the annotated region and weaker per-image AP than the larger reference. A representation or spatial-resolution bottleneck is a hypothesis; the present comparison does not identify its cause.
3. **Changed-domain false responses and region losses.** The AeBAD normal/background and fracture sheets illustrate the endpoint comparison. The FPR and region-transition tables quantify these errors. The independently scaled heatmaps show their locations, but cannot establish higher scores or threshold errors on their own. A targeted new-domain adaptation with separate normal calibration is a possible research direction, but the tested interventions have not established a successful fix.

The visual commentary was prepared with AI assistance and has not been independently verified. The released defect masks remain the reference annotations. Keep image attribution and identify any crop or rescaling.

## Research limitations

We completed a controlled comparison, a negative compact-fusion ablation and a failure analysis on AeBAD. Multi-scale flow architectures and domain-shift false alarms are already studied. A stronger research paper would require a concrete unresolved question, a valid comparator, and an independently informative result. More threshold trials on the same development defects do not provide that.

The normalization, alignment and fusion pilots did not establish a successful fix. These were exploratory pilots, separate from the planned three-seed benchmark.
