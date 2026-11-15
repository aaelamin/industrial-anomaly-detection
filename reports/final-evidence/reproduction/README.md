# Frozen experiment reproduction suite

This suite gives one configured entry point for the seven MVTec experiment groups and the AeBAD development experiment. It uses the original experiment sources and settings. It creates a separate output workspace; it never resumes the authors' historical result directories.

## Status and scope

The copied legacy sources match the completed experiments' source receipts. The additional four-route runner previously passed exact normal-only recovery checks on the authors' Windows/CUDA machine. This combined suite is a new orchestration candidate: see `QUALIFICATION-STATUS.json` when supplied for its own fresh-output test outcome. A successful short bootstrap does **not** establish a clean-machine installation or a complete second execution of the full study.

The scientific study is an adaptation and controlled comparison. This package does not establish a novel anomaly detector or reproduce every setting in the original FastFlow publication.

## Setup

Use Windows with an NVIDIA GPU. The executed environment uses Python 3.12.13 and the exact versions in `additional/requirements-observed.txt`; availability of those versions in a new environment has not been tested. Install matching PyTorch CUDA wheels, then the remaining pinned dependencies in an isolated environment. `psutil==7.0.0` is required for owned-process cleanup. Do not alter an environment used by an active study.

Obtain MVTec AD and AeBAD from their providers under their data terms. Dataset images and pretrained weights are not included. The relative membership and SHA-256 hashes in `additional/manifest.json` identify the permitted inputs. The 1,479 reserved AeBAD images are excluded. Do not replace the manifest with a broader test set.

Copy `host-config.example.json` to a private `host-config.json` and replace every placeholder with an absolute path. Keep the package, dataset directories and output workspace separate. Use an empty workspace on first execution. Provide the three exact weight files shown in the example; also place the ResNet and WideResNet files under `<torch_home>/hub/checkpoints/` with their original filenames. Weight hashes are verified, and this runner does not download replacements. `extra_pythonpath` is optional for an explicitly maintained dependency directory; normally install dependencies in the environment instead.

The legacy Week 5 source expects a fixed relative dataset path. Setup creates a Windows directory junction inside the **new workspace** pointing at the configured MVTec directory. The dataset remains read-only. Windows may require permission to create that junction. All logs and results remain under the configured workspace. The shared GPU lock prevents concurrent runs that use that same lock; it is not a system-wide lock against unrelated applications.

## Commands

From this package directory, using the configured Python environment:

```powershell
python -B reproduce.py plan --config host-config.json
python -B reproduce.py preflight --config host-config.json
python -B reproduce.py bootstrap --config host-config.json --wall-hours 1
```

`plan` checks source identities and lists the 318 planned jobs without loading models or images. `preflight` additionally checks package versions and weight hashes. `bootstrap` runs bounded implementation checks in fresh output folders: original ResNet recovery, original WideResNet recovery, the original compact-flow smoke test, and four additional normal-only routes. It intentionally crashes before a checkpoint or memory-bank commit, verifies the actual saved state, resumes in a fresh process, compares state and normal predictions bit for bit, and verifies completed runs are no-ops.

The original ResNet, WideResNet and compact-flow checks use all 209 bottle training normals, matching their original source membership. The compact-flow smoke test additionally uses the already declared 83 bottle test images and masks. Those scores are engineering checks, not new paper results. The four additional diagnostic routes use the previously frozen 65-normal bottle subset (the PatchCore diagnostic uses the first eight). The AeBAD-recipe diagnostic also uses bottle normals. No reserved AeBAD input is needed for bootstrap.

For an explicitly authorized full reproduction, the one entry point is:

```powershell
python -B reproduce.py reproduce --config host-config.json --wall-hours 8
```

This verifies or performs the bootstrap first, then executes the declared study arms. The full study takes multiple bounded sessions; eight hours is a session cap, not a completion estimate. Rerun the same command with the same package and workspace to continue. Completed result receipts are verified and skipped. Uncommitted work resumes from the source runner's checkpoints. The bootstrap has one persisted one-hour clock; retrying does not reset that clock. A failed or expired bootstrap requires diagnosis and a documented new qualification, rather than deleting its receipt or quietly bypassing it.

Startup requires 5 GiB available host RAM, 8 GiB free disk and 4 GiB free GPU memory. These are startup floors, not estimates of the total disk required for 318 retained jobs. Monitor remaining disk during a full reproduction; exact historical results can consume substantially more. `pause_paths` stops new work and causes the supervisor to stop its owned subprocesses. The wrapper enforces a wall limit and holds a Windows idle-sleep request while active. It cannot prevent a power loss, restart, OS update or the computer being shut down.

## Planned result groups

| Group | Categories / seeds | Jobs |
|---|---|---:|
| Original ResNet-18 flow | 15 / 3 | 45 |
| Original WideResNet-50 flow | 15 / 3 | 45 |
| Compact middle-layer and multi-scale fusion flows | 15 / 3 / 2 arms | 90 |
| EfficientNet-B4 flow | 15 / 3 | 45 |
| Matched-recipe ResNet-18 flow | 15 / 3 | 45 |
| PatchCore reference | 15 / 3 | 45 |
| AeBAD development-domain flow | 1 / 3 | 3 |

The original compact-flow code also checks an archived 18-run engineering-pilot receipt. That receipt is preserved and labelled historical in `suite-manifest.json`; the fresh bootstrap separately repeats the original two-epoch smoke/recovery check. It does not silently treat the archived pilot as a new execution.

Dataset checksums are verified before use, accesses are restricted to declared roles, and access receipts are retained. Original legacy source cleanup is limited to its newly materialized workspace. Historical results, source datasets and external checkpoints are not cleanup targets.

## Provenance and interpretation

`suite-manifest.json`, `suite-lock.json`, `legacy-provenance.json`, and `additional/release-lock.json` bind source identities. A change to a locked input requires an explicitly versioned candidate and a new qualification; do not edit the lock to make a changed result appear compatible. `suite-logs/`, `bootstrap-checks/`, `legacy-access/`, and the result stores preserve execution evidence.

Report all 15 categories and three seeds. Average categories equally within each seed, then report mean and sample standard deviation across the three seed means. Do not choose epochs, thresholds or excluded categories after seeing scores. The saved-table verifier checks existing numbers without rerunning the models.

Refer to `additional/THIRD_PARTY_NOTICES.txt` and the included anomalib license. Cite FastFlow, PatchCore, MVTec AD, AeBAD, and relevant component papers in the report. Data-provider restrictions apply to any shared heatmap examples.
