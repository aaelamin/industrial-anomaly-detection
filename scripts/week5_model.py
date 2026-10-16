"""Projected CNN features and a shared-size flow for the Week 5 comparison."""
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['MPLBACKEND'] = 'Agg'
from run import torch, np
from anomalib.models.image.fastflow.torch_model import create_fast_flow_block
from anomalib.models.image.fastflow.anomaly_map import AnomalyMapGenerator
from anomalib.models.image.fastflow.loss import FastflowLoss
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


class ProjectedFlow(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        shape = [config['channels'], config['grid'], config['grid']]
        self.norm = torch.nn.LayerNorm(shape)
        self.flow = create_fast_flow_block(shape, config['conv3x3_only'], config['hidden_ratio'], config['flow_steps'])
        self.map_generator = AnomalyMapGenerator((256, 256))

    def forward(self, features):
        z, jacobian = self.flow(self.norm(features))
        return z, jacobian

    def anomaly_map(self, features):
        z, _ = self(features)
        return self.map_generator([z])


def fit_projection(features, dataset, config):
    if not dataset.paths or any(p.parent.name != 'good' or p.parent.parent.name != 'train' for p in dataset.paths):
        raise ValueError('PCA fitting accepts only the normal training split.')
    levels = []
    for i, source in enumerate(features):
        n, c, h, w = source.values.shape
        if n != len(dataset):
            raise ValueError('Feature count does not match training paths.')
        rng = np.random.default_rng(config['projection_seed'] + i)
        indices = np.sort(rng.choice(n*h*w, min(config['pca_sample_limit'], n*h*w), replace=False))
        sample = np.asarray(source.values[indices//(h*w), :, (indices%(h*w))//w, indices%w], dtype=np.float32)
        components = 96 if i == 1 else 32
        with threadpool_limits(limits=4):
            pca = PCA(n_components=components, svd_solver=config['pca_solver'],
                      iterated_power=config['pca_power_iterations'], random_state=config['projection_seed'])
            pca.fit(sample)
            projected = (sample-pca.mean_) @ pca.components_.T
        offset = projected.mean(axis=0)
        std = projected.std(axis=0)
        levels.append({'mean': torch.from_numpy(pca.mean_.copy()),
            'components': torch.from_numpy(pca.components_.copy()), 'offset': torch.from_numpy(offset),
            'scale': torch.from_numpy(np.maximum(std, config['scale_floor'])),
            'floor_count': int((std < config['scale_floor']).sum()),
            'sample_indices': torch.from_numpy(indices), 'source_shape': [n,c,h,w]})
    return {'levels': levels, 'training_paths': [p.relative_to(dataset.root).as_posix() for p in dataset.paths],
            'sampling_seed': config['projection_seed'], 'fit_split': 'train/good'}


class Projection(torch.nn.Module):
    def __init__(self, saved, grid=64):
        super().__init__()
        self.grid = grid
        for i, p in enumerate(saved['levels']):
            self.register_buffer(f'weight_{i}', p['components'][:,:,None,None])
            self.register_buffer(f'bias_{i}', -p['components'] @ p['mean'])
            self.register_buffer(f'offset_{i}', p['offset'][None,:,None,None])
            self.register_buffer(f'scale_{i}', p['scale'][None,:,None,None])

    def forward(self, features, arm):
        if arm not in ('middle','fusion'):
            raise ValueError(arm)
        outputs = []
        for i in ([1] if arm == 'middle' else [0,1,2]):
            width = 96 if arm == 'middle' else 32
            value = torch.nn.functional.conv2d(features[i], getattr(self,f'weight_{i}')[:width], getattr(self,f'bias_{i}')[:width])
            value = (value-getattr(self,f'offset_{i}')[:,:width]) / getattr(self,f'scale_{i}')[:,:width]
            outputs.append(torch.nn.functional.interpolate(value, size=(self.grid,self.grid), mode='bilinear', align_corners=False))
        return torch.cat(outputs, dim=1)


def train_epoch(model, features, optimizer, generator, config):
    model.train()
    order = torch.randperm(len(features), generator=generator)
    batch_size = config['batch_size']
    losses, gradients, gaussian_terms, jacobians = [], [], [], []
    for start in range(0, len(order)//batch_size*batch_size, batch_size):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = gaussian_sum = jacobian_sum = 0.0
        for indices in order[start:start+batch_size].split(config['micro_batch_size']):
            values = features[indices].to(next(model.parameters()).device)
            z, jacobian = model(values)
            loss = FastflowLoss()([z], [jacobian])
            if not torch.isfinite(loss):
                raise RuntimeError('Non-finite training loss.')
            fraction = len(indices)/batch_size
            (loss*fraction).backward()
            loss_sum += float(loss.detach())*fraction
            gaussian_sum += float((.5*z.detach().square().sum((1,2,3))).mean())*fraction
            jacobian_sum += float(jacobian.detach().mean())*fraction
        norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in model.parameters() if p.requires_grad]))
        if not torch.isfinite(norm):
            raise RuntimeError('Non-finite gradient norm.')
        gradients.append(float(norm))
        optimizer.step()
        losses.append(loss_sum)
        gaussian_terms.append(gaussian_sum)
        jacobians.append(jacobian_sum)
    if not losses:
        raise ValueError('Training set is smaller than the effective batch.')
    return {'loss': float(np.mean(losses)), 'max_gradient_norm': max(gradients),
            'mean_gradient_norm': float(np.mean(gradients)), 'gaussian_term': float(np.mean(gaussian_terms)),
            'log_jacobian': float(np.mean(jacobians)), 'steps': len(losses)}
