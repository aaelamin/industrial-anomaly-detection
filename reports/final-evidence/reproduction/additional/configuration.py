"""Explicit paths and content identities for the additional-study release candidate."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def relative(root, value):
    if not isinstance(value, str) or not value or '\\' in value or ':' in value:
        raise ValueError('Expected a nonempty portable relative path')
    p = PurePosixPath(value)
    if p.is_absolute() or '..' in p.parts or str(p) != value or value == '.':
        raise ValueError('Noncanonical or escaping relative path')
    root = Path(root).resolve()
    out = root.joinpath(*p.parts).resolve()
    if out == root or not out.is_relative_to(root):
        raise ValueError('Resolved path escapes root')
    return out


def disjoint(roots):
    values = list(roots.items())
    for i, (name, root) in enumerate(values):
        for other, target in values[i+1:]:
            if root.is_relative_to(target) or target.is_relative_to(root):
                raise ValueError(f'Overlapping roots: {name}, {other}')


class Configuration:
    def __init__(self, package, config):
        self.package = Path(package).resolve()
        self.manifest = read(self.package / 'manifest.json')
        self.host = read(config)
        if self.manifest['schema'] != 1 or set(self.host['data_roots']) != {'mvtec', 'aebad'}:
            raise ValueError('Unsupported manifest or dataset roots')
        self.roots = {k: Path(v).resolve() for k, v in self.host['data_roots'].items()}
        self.output = Path(self.host['output_root']).resolve()
        self.scratch = Path(self.host['scratch_root']).resolve()
        self.gpu_lock = Path(self.host['gpu_lock']).resolve()
        disjoint({**self.roots, 'output': self.output, 'scratch': self.scratch, 'package': self.package})
        if any(self.gpu_lock.is_relative_to(r) for r in self.roots.values()):
            raise ValueError('GPU lock must not be in a dataset')
        self.weights = {k: Path(v).resolve() for k, v in self.host['weight_paths'].items()}
        if set(self.weights) != set(self.manifest['weights']):
            raise ValueError('Provide exactly the declared weight identities')
        self._check_roles()

    def _check_roles(self):
        for dataset, sources in [('mvtec', self.manifest['datasets']['mvtec'].values()),
                                 ('aebad', [self.manifest['datasets']['aebad']])]:
            for source in sources:
                images = set()
                for role in ['train', 'calibration', 'evaluation']:
                    local = set()
                    for row in source[role]:
                        path = relative(self.roots[dataset], row['path'])
                        if row['dataset'] != dataset or len(row['sha256']) != 64:
                            raise ValueError('Invalid dataset identity')
                        if path in images or path in local:
                            raise ValueError('Duplicate/overlapping image roles')
                        if role in ('train', 'calibration') and (row['label'] != 0 or row['defect'] != 'good'):
                            raise ValueError('Normal-only fitting/calibration violated')
                        local.add(path)
                    images.update(local)
                masks = {r['path'] for r in source['masks']}
                if len(masks) != len(source['masks']):
                    raise ValueError('Duplicate mask')
                for row in source['masks']:
                    relative(self.roots[dataset], row['path'])
                for row in source['evaluation']:
                    if row.get('mask') and row['mask'] not in masks:
                        raise ValueError('Missing declared mask')
        allowed = {r['path']:r for r in self.manifest['datasets']['mvtec']['bottle']['train']}
        for row in self.manifest['diagnostic']['normal_rows']:
            if row != allowed.get(row['path']) or row['label'] != 0:
                raise ValueError('Diagnostic data must be declared bottle training normals')

    def resolved_row(self, row):
        out = dict(row)
        root = self.roots[row['dataset']]
        out['path'] = str(relative(root, row['path']))
        if row.get('mask'):
            out['mask'] = str(relative(root, row['mask']))
        return out

    def resolved_job(self, job):
        out = dict(job)
        out['weights'] = str(self.weights[job['weights_sha256']])
        return out

    def canonical(self, value):
        if isinstance(value, dict):
            return {k:self.canonical(v) for k,v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.canonical(v) for v in value]
        if isinstance(value, str) and Path(value).is_absolute():
            p = Path(value).resolve()
            for h, path in self.weights.items():
                if p == path:
                    return {'weight_sha256':h}
            for name, root in {**self.roots, 'output':self.output, 'scratch':self.scratch, 'package':self.package}.items():
                if p.is_relative_to(root):
                    return {'root':name, 'relative':p.relative_to(root).as_posix()}
            raise ValueError('Unmapped absolute path in scientific identity')
        return value

    def fingerprint(self):
        return digest({'manifest':sha(self.package/'manifest.json'),
                       'release_lock':read(self.package/'release-lock.json')})

    def verify_sources(self, packages=True):
        expected = {**self.manifest['source_files'], **read(self.package/'release-lock.json')['files']}
        for name, h in expected.items():
            if sha(relative(self.package, name)) != h:
                raise ValueError('Release source changed: ' + name)
        if packages:
            for name, v in self.manifest['packages'].items():
                if importlib.metadata.version(name) != v:
                    raise ValueError('Installed package version differs: ' + name)
        return {'manifest_sha256':sha(self.package/'manifest.json'), 'source_files':expected,
                'packages':self.manifest['packages'], 'release_identity':self.fingerprint()}

    def verify_inputs(self, normal_only=True, job_id=None):
        if normal_only:
            rows = self.manifest['diagnostic']['normal_rows']
            jobs = self.manifest['diagnostic']['jobs']
        else:
            rows = [r for source in [self.manifest['datasets']['aebad'], *self.manifest['datasets']['mvtec'].values()]
                    for role in ['train','calibration','evaluation','masks'] for r in source[role]]
            jobs = self.manifest['jobs']
        if job_id is not None:
            jobs = [j for j in jobs if j['id'] == job_id]
            if not jobs:
                raise ValueError('Unknown job')
            if normal_only and jobs[0]['kind'] == 'patchcore':
                rows = rows[:8]
            elif not normal_only:
                j = jobs[0]
                src = self.manifest['datasets']['aebad'] if j['dataset']=='aebad' else self.manifest['datasets']['mvtec'][j['category']]
                rows = [r for role in ['train','calibration','evaluation','masks'] for r in src[role]]
        for row in rows:
            p = relative(self.roots[row['dataset']], row['path'])
            if sha(p) != row['sha256']:
                raise ValueError('Input content differs: ' + row['path'])
        for h in {j['weights_sha256'] for j in jobs}:
            if sha(self.weights[h]) != h:
                raise ValueError('Weight content differs')
        return {'input_hashes':len(rows), 'weights':len({j['weights_sha256'] for j in jobs}),
                'normal_only':normal_only, 'reserved_images_opened':0}

    def prepare_store(self):
        identity = self.fingerprint()
        marker = self.output / 'release-identity.json'
        if self.output.exists() and any(self.output.iterdir()):
            if not marker.exists() or read(marker).get('release_identity') != identity:
                raise ValueError('Refuse nonempty legacy or different-identity output store')
        self.output.mkdir(parents=True, exist_ok=True)
        if not marker.exists():
            marker.write_text(json.dumps({'release_identity':identity},indent=2)+'\n', encoding='utf-8')
        return identity
