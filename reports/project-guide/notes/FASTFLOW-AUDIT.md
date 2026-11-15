# FastFlow released-code comparison

The completed ResNet and WideResNet configurations are close to the community release's final-epoch **pixel AUROC** aggregates. Whole-image detection remains below the paper. These are different comparisons.

| Comparison | Our mean ± sample SD | Published comparison |
| --- | --- | --- |
| Original ResNet pixel AUROC, final endpoint | 95.41 ± 0.59 | Community release final: 95.6 |
| WideResNet pixel AUROC, final endpoint | 97.19 ± 0.49 | Community release final: 97.2 |
| WideResNet image AUROC | 93.17 ± 2.14 | Paper without augmentation: 98.9 |
| WideResNet pixel AUROC | 97.19 ± 0.49 | Paper without augmentation: 98.2 |

The [community implementation](https://github.com/gathierry/FastFlow/tree/2cf1f2f4c562a7f13cfb1959e3afe5df2f2d2565) explicitly calls itself unofficial. Its final and best-epoch columns use different endpoint policies; our primary endpoint is fixed at 500 epochs. Its table evaluates pixels, not whole-image detection. Proximity of rounded aggregates is not a statistical equivalence result. See the [FastFlow paper](https://arxiv.org/abs/2111.07677) and our [complete tables](../../final-evidence/team-tables/README.md).

Shared-state constructed float32 checks found agreement in the core flow equations, loss, gradients and full-batch optimizer update. Independently initialized models do not match at the same integer seed because the releases draw random permutations differently. A strict microbatch-update case and double-precision loss accumulation check failed their frozen tolerances. Those limits were retained; they do not establish the cause of the benchmark gap. The audit did not rerun the original dependency environment or compare entire production training trajectories.

Matching ResNet's flow-filter pattern to WideResNet did not close the gap: pixel AP was 39.11 ± 0.54 versus 39.69 ± 1.04 for original ResNet. Pretraining and remaining recipe differences limit a purely causal comparison of the backbones. These results describe the tested adaptations; they do not reproduce every headline score.

[Project guide](../README.md) · [Method details](../../final-evidence/notes/METHODS.md)
