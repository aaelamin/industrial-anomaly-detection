# Inputs for a fresh reproduction

Use the delivered manifest as the authority for membership and file identity. Dataset roots in the private host configuration are user settings; the scripts do not need the original computer's paths. The runner joins each root with the relative path recorded in `reproduction/additional/manifest.json`.

## Datasets

Obtain the original 15-category MVTec AD from the [provider](https://www.mvtec.com/research-teaching/datasets/mvtec-ad), following its download and CC BY-NC-SA 4.0 terms. This study does not use MVTec AD 2. Under the configured MVTec root, the layout is `<category>/train/good/`, `<category>/test/<defect>/` and `<category>/ground_truth/<defect>/`. The declared split contains 3,629 training images and 1,725 test images. Preserve original images and names; re-encoding changes the manifest hashes.

Obtain AeBAD through the download links in the authors' [official release](https://github.com/zhangzilongc/MMR). The dataset is CC BY 4.0. The manifest's paths start with `AeBAD/AeBAD_S/`, so the configured AeBAD root must be the directory **containing `AeBAD`**, not the innermost `AeBAD_S` directory. For example, a calibration input resolves as `<configured root>/AeBAD/AeBAD_S/train/good/view/IMG_6992.png`. Check this join before running; folder names used in the host-config example are placeholders.

Only the listed 128 fitting normals, 32 calibration normals, 160 development images and their 128 defect masks belong to this AeBAD study. The evaluator uses manifest membership rather than discovering more test files. The 1,479 reserved images are excluded. The original MMR training experiment and other exploratory pilots are not required to reproduce this final FastFlow comparison.

## Exact pretrained artifacts

These are the weight files used in the experiments. Check the hashes after downloading; a different set of weights defines a different configuration. Download availability has not been rechecked for this documentation update.

| Backbone | Recorded artifact | SHA-256 |
|---|---|---|
| ResNet-18 | [resnet18-5c106cde.pth](https://download.pytorch.org/models/resnet18-5c106cde.pth) | `5c106cde386e87d4033832f2996f5493238eda96ccf559d1d62760c4de0613f8` |
| WideResNet-50-2 | [wide_resnet50_racm-8234f177.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/wide_resnet50_racm-8234f177.pth) | `8234f177589d8239cc335ec7f23d69c8c7a323f9a0ba0f9df830ccb66a6f83fa` |
| EfficientNet-B4 | [efficientnet_b4_ra2_320-7eb33cd5.pth](https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/efficientnet_b4_ra2_320-7eb33cd5.pth) | `7eb33cd5c8208bac4f23f2f8e61df12e4cf977ed1f4aadcc1ec4d59ffaf2d127` |

Set each `weight_paths` entry to the corresponding file. The original legacy loaders also expect the ResNet and WideResNet files in `<torch_home>/hub/checkpoints/` under the filenames above. The hashes are the same in both locations. Existing caches can be reused; do not duplicate large weights unnecessarily. The suite verifies hashes before use and keeps network model downloads disabled.

## What is and is not distributed

The repository contains source code, configurations, relative input paths, metric tables, attributed overlays and verification records. It excludes datasets, pretrained weights, full prediction arrays, personal host configurations and trained checkpoints. The configured reproduction command retrains the declared models into a fresh output workspace; it is not a quick replay of saved predictions. The data and model providers' terms still apply.

Use the pinned setup and bounded bootstrap before a full run. Keep its actual qualification status with the report's reproducibility claim. A successful check on the original machine does not establish a clean install on another machine, nor does it prove that all full runs have been independently repeated.
