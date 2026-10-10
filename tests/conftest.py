"""Shared test helpers.

`orbit_reference` is a short reference implementation of the rspace3d 2.2.0
symmetrisation (np.nanmedian and np.nanmean over the orbit members) with the
variance rule of rspace3d 2.3. The orbit engine must give the same result.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from rspace3d.volume_builder import _EDGE_SNAP_TOL, index_space_ops


def _coords(op, shape):
    """float64 source coordinate of each voxel, ((m0 c0 + m1 c1) + m2 c2) + t."""
    c = np.indices(shape).reshape(3, -1).astype(np.float64)
    M = np.asarray(op['M'], np.float64)
    t = np.asarray(op['t'], np.float64)
    return np.stack([((M[d, 0] * c[0] + M[d, 1] * c[1]) + M[d, 2] * c[2]) + t[d]
                     for d in range(3)])


def _member(o, op, data, variance):
    """Value, variance and source key of one orbit member for every voxel."""
    shape = data.shape
    pos = _coords(op, shape)
    if op['kind'] != 'interp':
        src = np.rint(pos).astype(np.int64)
        ok = np.all((src >= 0) & (src < np.array(shape)[:, None]), axis=0)
        key = np.ravel_multi_index(tuple(np.where(ok, src, 0)), shape)
        x = data.reshape(-1)[key]
        x = np.where(ok & np.isfinite(x), x, np.nan).astype(np.float32)
        xv = None if variance is None else variance.reshape(-1)[key]
        return x, xv, key
    # trilinear interpolation of the measured neighbours (rspace3d 2.2)
    for d, n in enumerate(shape):
        row = pos[d]
        row[(row < 0) & (row > -_EDGE_SNAP_TOL)] = 0.0
        row[(row > n - 1) & (row < n - 1 + _EDGE_SNAP_TOL)] = n - 1
    finite = np.isfinite(data)
    num = map_coordinates(np.where(finite, data, 0).astype(np.float32), pos, order=1,
                          mode='constant', cval=0.0)
    den = map_coordinates(finite.astype(np.float32), pos, order=1, mode='constant', cval=0.0)
    with np.errstate(invalid='ignore', divide='ignore'):
        x = np.where(den > 1e-6, num / np.maximum(den, 1e-6), np.nan).astype(np.float32)
    inside = np.all((pos >= 0) & (pos <= np.array(shape)[:, None] - 1), axis=0)
    fl = np.floor(pos)
    frac = pos - fl
    point = np.all(frac == 0, axis=0)
    key = np.where(point & inside, np.ravel_multi_index(
        tuple(np.where(inside, fl, 0).astype(np.int64)), shape), -1 - o)
    xv = None
    if variance is not None:
        # sum_k (w_k / W)^2 var_k over the measured corners k
        w2v = np.zeros(pos.shape[1])
        wsum = np.zeros(pos.shape[1])
        base = np.where(inside, fl, 0).astype(np.int64)
        for s in np.ndindex(2, 2, 2):
            w = np.ones(pos.shape[1])
            idx = []
            for d in range(3):
                w = w * (frac[d] if s[d] else 1.0 - frac[d])
                idx.append(np.minimum(base[d] + s[d], shape[d] - 1))
            flat = np.ravel_multi_index(tuple(idx), shape)
            use = inside & (w > 0) & np.isfinite(data.reshape(-1)[flat])
            wsum += np.where(use, w, 0.0)
            w2v += np.where(use, w * w * variance.reshape(-1)[flat], 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            xv = w2v / (wsum * wsum)
    return x, xv, key


def reference_symmetrize(vol, group, sigma=3.0, min_valid=3, poisson_floor=True,
                         variance=None):
    """rspace3d 2.2.0 algorithm. Returns (intensity, variance, n_outliers_removed).

    `variance` is the input variance array (or None). The output variance is
    sum_j (c_j / m)^2 var_j over the distinct sources j of the m survivors.
    """
    ops = index_space_ops(vol, group)
    ident = next(i for i, o in enumerate(ops) if np.array_equal(o['op'], np.eye(3, dtype=int)))
    data = np.asarray(vol.intensity, np.float32)
    members = [_member(i, o, data, variance) for i, o in enumerate(ops)]
    E = np.stack([m[0] for m in members])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        n_removed = 0
        if sigma is not None:
            n_valid = np.isfinite(E).sum(axis=0)
            med = np.nanmedian(E, axis=0)
            dev = np.abs(E - med[None])
            scale = np.nanmedian(dev, axis=0) * 1.4826
            if poisson_floor:
                scale = np.maximum(scale, np.sqrt(np.maximum(med, 1.0)))
            outlier = (dev > sigma * scale[None]) & (n_valid[None] >= min_valid)
            n_removed = int(outlier[ident].sum())
            E = np.where(outlier, np.float32(np.nan), E)
        out = np.nanmean(E, axis=0).astype(np.float32)
    out_var = None
    if variance is not None:
        surv = np.isfinite(E)
        keys = np.stack([m[2] for m in members])
        V = np.stack([m[1] for m in members]).astype(np.float64)
        m = surv.sum(axis=0)
        out_var = np.zeros(E.shape[1])
        for i in range(len(ops)):
            mult = np.zeros(E.shape[1])
            for j in range(len(ops)):
                mult += surv[j] & (keys[j] == keys[i])
            out_var += np.where(surv[i], mult * V[i], 0.0)
        with np.errstate(invalid='ignore', divide='ignore'):
            out_var = np.where(m > 0, out_var / (m * m.astype(np.float64)), np.nan)
        out_var = out_var.reshape(data.shape)
    return out.reshape(data.shape), out_var, n_removed


@pytest.fixture
def orbit_reference():
    return reference_symmetrize
