# Additional-study runner candidate

Four normal-only GPU equivalence and recovery routes passed on the development machine. The suite controller has separate qualification records. A full reproduction on a clean machine remains untested. Execution requires a matching qualification receipt for the selected route; saved-metric verification does not meet that requirement.

The package contains byte-identical scientific worker, feature-adapter, evaluation and region-metric files from the frozen additional study. The separate configuration/common-module adapter supplies explicit paths, content identities, resource limits and owned-process cleanup. The original study and its checkpoints are not changed. Original Week 3/4/5 compact/PCA commands remain separate; this candidate is not a one-command reproduction of every paper result.

## Verified without models

The CPU checks passed source-package/current-directory relocation, unsafe relative paths and junction escape rejection, disjoint roots, source/manifest tampering, missing or changed input bytes, changed weight bytes, legacy-output refusal and different-identity resume refusal. A private normal-input preflight also verified the 65 declared bottle training normals and three weight files. No model runtime was imported for these checks.

The manifest contains content hashes for 7,060 already allowed MVTec/AeBAD image and mask files. This export decoded no images and opened no reserved AeBAD file. Historical MVTec inputs used size/mtime identities; the new SHA-256 export attests their current content, not a retroactive hash check. The 1,479 reserved AeBAD images are absent from the manifest. Path/hash separation does not establish physical-specimen independence.

## Configure a separate store

Copy `host-config.example.json` outside this package, replace every placeholder with an absolute path on your machine, and keep that private configuration out of Git. Dataset roots, output, scratch and this package must not overlap. Specify all three verified pretrained weight files. Weights and datasets are not included or downloaded automatically.

`gpu_lock` must be the same shared lock path used by any other research process on that GPU. Never launch this candidate alongside another study using a different lock. The normal diagnostic uses only the 65 listed bottle training normals; the PatchCore route uses the first eight. The all-3x3 route is tested on those same bottle normals, not on AeBAD defects.

From any directory, using the path to this package:

```text
python runner.py preflight --config PRIVATE_CONFIG --metadata-only
python runner.py preflight --config PRIVATE_CONFIG
```

The first command checks paths and source identities without dataset reads, model imports or output creation. The second also checks the observed installed package versions, normal-image hashes and weight hashes. Its PASS means input readiness only.

## Normal-only qualification interface

Use Windows and the CUDA environment recorded in `requirements-observed.txt`. That file records the observed environment; installation of every exact version on a new machine has not been tested. The candidate requires 3 GiB available system RAM, 8 GiB free disk and 4 GiB free GPU memory at launch. It retains the resource waits and disk-backed feature cache; it does not close applications or change power settings.

An explicit, separately bounded test can use:

```text
python runner.py normal-check --config PRIVATE_CONFIG --job diagnostic-resnet --branch control --wall-minutes 10
python runner.py normal-check --config PRIVATE_CONFIG --job diagnostic-resnet --branch interrupted --inject-crash --wall-minutes 10
python runner.py normal-check --config PRIVATE_CONFIG --job diagnostic-resnet --branch interrupted --wall-minutes 10
```

The injected process must exit with code 91 before the second-epoch checkpoint commit. The last command resumes the saved first epoch. Other declared IDs are `diagnostic-efficientnet`, `diagnostic-aebad` and `diagnostic-patchcore`. These commands alone do not certify equality: compare final tensors, optimizer/RNG/history or memory bank, normal probes, actual interruption and completed no-op behavior against the frozen local controls before producing a qualification receipt. Do not run this GPU phase concurrently with the current study.

The `run` interface is receipt-gated. Its four normal-only routes passed on the development machine; fresh-machine production remains unverified. It uses the fixed 500-epoch endpoint and declared evaluation roles, with at most a one-hour launch before resuming the same job. Its existence is not evidence of successful production execution. No new full study was launched through it. A suite controller and the public one-command reproduction requirement remain open.

## Limits and attribution

Source-package relocation was tested on the original Windows machine. Clean-machine installation, CPU training, other operating systems, full-run equivalence and relocation of already-written output indexes remain unverified. Do not move historical checkpoints into this new output identity. Checkpoint compatibility is stricter than sharing a model name.

Dataset imagery, weights, source score maps, private configuration and historical checkpoints are excluded from this package. MVTec AD: Bergmann et al. (2019), CC BY-NC-SA 4.0, https://www.mvtec.com/research-teaching/datasets/mvtec-ad . AeBAD-S: Zhang et al. (2023), CC BY 4.0, https://github.com/zhangzilongc/MMR . Obtain data through the original providers and retain their terms. Metadata does not grant a new dataset license.

The feature adapter retains its Anomalib/FastFlow Apache-2.0 attribution in the source header. `THIRD_PARTY_NOTICES.txt` identifies the adaptation, and `LICENSE-anomalib.txt` is the verbatim license from the installed Anomalib distribution. Installed third-party packages retain their own licenses. The other staged research files belong to the project. This package does not assign a blanket license to those files.
