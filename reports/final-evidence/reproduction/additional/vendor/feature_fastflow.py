"""CNN adapter retaining installed Anomalib FastFlow's flow and scoring path.

Adapted constructor based on anomalib 2.6.2 FastflowModel (Apache-2.0,
Copyright 2022 gathierry; 2022-2025 Intel Corporation). No pretrained
network access occurs in this module.
"""
from pathlib import Path
import hashlib
import timm
import torch
from torch import nn
from anomalib.models.image.fastflow.torch_model import FastflowModel, create_fast_flow_block
from anomalib.models.image.fastflow.anomaly_map import AnomalyMapGenerator


class FeatureFastFlow(FastflowModel):
    """Use the same CNN forward path with explicit timm feature metadata."""

    def __init__(self, backbone='efficientnet_b4.ra2_in1k', input_size=(256, 256),
                 flow_steps=8, conv3x3_only=False, hidden_ratio=1.0):
        nn.Module.__init__(self)
        if backbone not in {'resnet18', 'wide_resnet50_2', 'efficientnet_b4.ra2_in1k'}:
            raise ValueError(f'Unaudited backbone: {backbone}')
        self.input_size = tuple(input_size)
        self.backbone_name = backbone
        self.feature_extractor = timm.create_model(backbone, pretrained=False,
                                                   features_only=True, out_indices=[1, 2, 3])
        self.feature_channels = list(self.feature_extractor.feature_info.channels())
        self.feature_strides = list(self.feature_extractor.feature_info.reduction())
        if self.feature_strides != [4, 8, 16]:
            raise ValueError(f'Unexpected feature strides: {self.feature_strides}')
        if any(s % r for s in self.input_size for r in self.feature_strides):
            raise ValueError('Input size must be divisible by each feature stride.')
        shapes = [[c, input_size[0] // r, input_size[1] // r]
                  for c, r in zip(self.feature_channels, self.feature_strides, strict=True)]
        self.norms = nn.ModuleList([nn.LayerNorm(s, elementwise_affine=True) for s in shapes])
        self.feature_extractor.requires_grad_(False).eval()
        self.fast_flow_blocks = nn.ModuleList([
            create_fast_flow_block(s, conv3x3_only, hidden_ratio, flow_steps) for s in shapes
        ])
        self.anomaly_map_generator = AnomalyMapGenerator(input_size=self.input_size)

    def train(self, mode=True):
        super().train(mode)
        self.feature_extractor.eval()
        return self

    def from_features(self, features):
        """Equivalent to reference forward after the frozen feature extractor."""
        if len(features) != len(self.norms):
            raise ValueError('Feature-level count mismatch.')
        latents, determinants = [], []
        for feature, norm, flow in zip(features, self.norms, self.fast_flow_blocks, strict=True):
            if tuple(feature.shape[1:]) != tuple(norm.normalized_shape):
                raise ValueError('Feature shape mismatch.')
            z, jac = flow(norm(feature))
            latents.append(z)
            determinants.append(jac)
        return latents, determinants

    def load_verified_backbone(self, path, sha256):
        path = Path(path)
        with path.open('rb') as stream:
            actual = hashlib.file_digest(stream, 'sha256').hexdigest()
        if actual != sha256:
            raise ValueError('Pretrained weight hash mismatch.')
        weights = torch.load(path, map_location='cpu', weights_only=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        expected = self.feature_extractor.state_dict()
        missing = set(expected) - set(weights)
        unused = set(weights) - set(expected)
        allowed_unused = ('conv_head.', 'bn2.', 'classifier.') if self.backbone_name.startswith('efficientnet') else ('layer4.', 'fc.')
        unexpected = [k for k in unused if not k.startswith(allowed_unused)]
        if missing or unexpected:
            raise ValueError(f'Invalid feature weights: missing={sorted(missing)}, unexpected={sorted(unexpected)}')
        self.feature_extractor.load_state_dict({k: weights[k] for k in expected}, strict=True)
        return {'sha256': actual, 'loaded_keys': len(expected), 'unused_head_keys': sorted(unused)}
