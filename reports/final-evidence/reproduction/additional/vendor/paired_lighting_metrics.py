"""Bounded-memory native-resolution metrics with explicit quantization bounds."""

import numpy as np
from scipy.ndimage import label
from sklearn.metrics import average_precision_score


class RegionMetrics:
    def __init__(self, bins=2**20):
        self.bins = bins
        self.positive = np.zeros(bins, dtype=np.int64)
        self.negative = np.zeros(bins, dtype=np.int64)
        self.region_mass = np.zeros(bins, dtype=np.float64)
        self.regions = 0
        self.images = 0

    def add(self, scores, mask):
        assert scores.shape == mask.shape
        assert np.isfinite(scores).all() and scores.min() >= -1e-6 and scores.max() <= 2 + 1e-6
        labels, regions = label(mask, structure=np.ones((3, 3)))
        counts = np.bincount(labels.ravel())
        indices = np.minimum((np.clip(scores, 0, 2) * (self.bins / 2)).astype(np.int32), self.bins - 1)
        positive_indices = indices[mask]
        positive = np.bincount(positive_indices, minlength=self.bins)
        self.positive += positive
        self.negative += np.bincount(indices.ravel(), minlength=self.bins) - positive
        if regions:
            weights = 1 / counts[labels[mask]]
            self.region_mass += np.bincount(positive_indices, weights=weights, minlength=self.bins)
        self.regions += regions
        self.images += 1

    def result(self, limit=0.05):
        used = (self.positive + self.negative) > 0
        pos = self.positive[used][::-1].astype(float)
        neg = self.negative[used][::-1].astype(float)
        mass = self.region_mass[used][::-1]
        assert pos.sum() > 0 and neg.sum() > 0 and self.regions > 0
        tp, fp = np.cumsum(pos), np.cumsum(neg)
        tp0, fp0 = tp - pos, fp - neg
        pro = np.cumsum(mass) / self.regions
        pro0 = pro - mass / self.regions
        fpr0, fpr = fp0 / neg.sum(), fp / neg.sum()
        width = np.maximum(0, np.minimum(fpr, limit) - np.minimum(fpr0, limit))
        full_width = fpr - fpr0
        portion = np.divide(width, full_width, out=np.zeros_like(width), where=full_width > 0)
        partial_pro = pro0 + (pro - pro0) * portion
        aupro = np.sum(width * (pro0 + partial_pro) / 2) / limit
        lower, upper = np.sum(width * pro0) / limit, np.sum(width * pro) / limit
        # All within-bin rankings and exact-score ties fall between these precision extrema.
        recall_steps = pos / pos.sum()
        ap = np.sum(recall_steps * tp / (tp + fp))
        ap_lower = np.sum(recall_steps * (tp0 + 1) / (tp0 + fp0 + neg + 1))
        ap_upper = np.sum(recall_steps * np.divide(tp, tp + fp0, out=np.ones_like(tp), where=(tp + fp0) > 0))
        return {'aupro_005': float(aupro), 'aupro_lower': float(lower), 'aupro_upper': float(upper),
                'pixel_ap': float(ap), 'pixel_ap_lower': float(ap_lower), 'pixel_ap_upper': float(ap_upper),
                'regions': self.regions, 'positive_pixels': int(pos.sum()), 'negative_pixels': int(neg.sum()),
                'images': self.images, 'score_bins': self.bins, 'score_bin_width': 2 / self.bins}


def brute_force(maps, masks, limit=0.05):
    labels_and_counts = [label(m, structure=np.ones((3, 3))) for m in masks]
    regions = sum(n for _, n in labels_and_counts)
    negative = sum((~m).sum() for m in masks)
    values = np.concatenate([s.ravel() for s in maps])
    x, y = [0.0], [0.0]
    for threshold in sorted(set(values), reverse=True):
        false = sum(((s >= threshold) & ~m).sum() for s, m in zip(maps, masks))
        overlaps = []
        for s, (labels, n) in zip(maps, labels_and_counts):
            overlaps.extend([float((s[labels == k] >= threshold).mean()) for k in range(1, n + 1)])
        x.append(false / negative)
        y.append(sum(overlaps) / regions)
    x, y = np.asarray(x), np.asarray(y)
    area = 0.0
    for j in range(1, len(x)):
        if x[j] == x[j-1] or x[j-1] >= limit:
            continue
        right = min(x[j], limit)
        end_y = y[j-1] + (y[j] - y[j-1]) * (right - x[j-1]) / (x[j] - x[j-1])
        area += (right - x[j-1]) * (y[j-1] + end_y) / 2
    return area / limit


def self_check():
    rng = np.random.default_rng(52)
    checks = []
    for trial in range(24):
        masks = [rng.random((9, 13)) < 0.12 for _ in range(3)]
        masks[0][:] = False
        maps = [rng.random(m.shape) * 2 for m in masks]
        if trial % 3 == 0:
            maps = [np.round(s * 4) / 4 for s in maps]
        if trial == 0:
            maps = [np.full(m.shape, 0.7) for m in masks]
        if trial == 1:
            maps = [m.astype(float) for m in masks]
        measure = RegionMetrics(bins=16 if trial < 12 else 2**20)
        for scores, mask in zip(maps, masks):
            measure.add(scores, mask)
        actual = measure.result()
        expected_pro = brute_force(maps, masks)
        expected_ap = average_precision_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([s.ravel() for s in maps]))
        assert actual['aupro_lower'] - 1e-12 <= expected_pro <= actual['aupro_upper'] + 1e-12, (trial, expected_pro, actual)
        assert actual['pixel_ap_lower'] - 1e-12 <= expected_ap <= actual['pixel_ap_upper'] + 1e-12, (trial, expected_ap, actual)
        # Independently evaluate the quantized scores and compare to the streaming implementation.
        quantized = [np.minimum((s * measure.bins / 2).astype(int), measure.bins-1).astype(float) for s in maps]
        quantized_pro = brute_force(quantized, masks)
        quantized_ap = average_precision_score(np.concatenate([m.ravel() for m in masks]), np.concatenate([s.ravel() for s in quantized]))
        assert abs(actual['aupro_005'] - quantized_pro) < 1e-12
        assert abs(actual['pixel_ap'] - quantized_ap) < 1e-12
        checks.append({'trial': trial, 'exact_aupro': expected_pro, 'exact_pixel_ap': float(expected_ap), 'bounds_passed': True})
    return {'passed': True, 'toy_checks': checks, 'connectivity': 8,
            'references': 'Brute-force unique thresholds for AU-PRO and sklearn average_precision_score. Includes constant scores, perfect maps, ties, disconnected regions and normal-only images.'}


if __name__ == '__main__':
    print(self_check())
