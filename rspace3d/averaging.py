"""
averaging.py — Noise model of the orbit members and scores of averaging options.

`symmetrize_volume` has three averaging options: `weighting` ('equal',
'counts', 'adaptive'), `rejection` (None, 'none', 'mad', 'model') and
`noise_model`. This module holds:

* `NoiseModel`: the variance of one orbit member.
* `gradient_magnitude` and `first_pass_gradient`: the gradient term G.
* `fit_noise_model`: a robust fit of the noise model to a volume.
* `automatic_candidates`, `score_candidates`, `bias_checks`, `suggest` and
  `preselect`: tools that compare averaging options on one volume and
  suggest one.

Noise model of member i of an orbit, at the orbit median med:

    v_i = gain max(med, 1) / N_i + (rel med)^2 + (kappa G)^2 + floor^2

* gain max(med, 1) / N_i: counting noise of a mean of N_i pixels. N_i is
  the pixel count of the member (`VolumeData.counts`, 1 without counts).
* (rel med)^2: scatter in proportion to the intensity (for example scale
  differences between frames).
* (kappa G)^2: the gradient guard. The members of an orbit sample a sharp
  feature (a Bragg peak, a superstructure peak, a diffuse rod) at different
  sub-voxel positions. Thus their scatter increases with the local gradient
  G. G is the gradient magnitude (intensity change per voxel) of a
  first-pass symmetric average at the target voxel (`first_pass_gradient`:
  equal weights, MAD rejection, the larger one-sided difference on each
  axis, thus G > 0 also at a peak maximum).
* floor^2: a constant variance.

The model uses the orbit median, not the value of the member. A variance
from the value of the member makes the weight of a low value too large, and
thus the weighted mean too low.

The functions use the NumPy engine of `orbit_engine` (`gather_members` and
`estimate`) on a sample of voxels. They do not use numba or the GPU, except
for the first pass and G on the full volume.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np
import numpy.typing as npt

from . import orbit_engine as oe

CHI2_MEDIAN = 0.4549364231195724     # median of the chi^2 distribution with 1 degree of freedom
_PARAMS = ('gain', 'rel', 'kappa', 'floor')
# floor^2 of a fit with gain = 0 and floor = 0, as a fraction of the median model variance
FLOOR_ADDED = 1e-6
NO_ORBITS = ("The noise-model fit found too few orbits with 3 or more members from different "
             "sources. The volume has no voxels with 3 or more finite orbit members that read "
             "different source voxels.")


# ──────────────────────────────────────────────────────────────────
# Noise model
# ──────────────────────────────────────────────────────────────────

@dataclass
class NoiseModel:
    """Variance of one orbit member (see the module docstring).

    All parameters are finite and not negative. `gain` or `floor` must be
    positive, so that each variance is positive. `info` holds details of a
    fit (JSON types only).
    """
    gain: float
    rel: float = 0.0
    kappa: float = 0.0
    floor: float = 0.0
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in _PARAMS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"NoiseModel.{name} must be finite and >= 0, got {value!r}")
            setattr(self, name, value)
        if self.gain <= 0 and self.floor <= 0:
            raise ValueError("NoiseModel needs gain > 0 or floor > 0")

    @property
    def params(self) -> tuple[float, float, float, float]:
        """(gain, rel, kappa, floor)."""
        return (self.gain, self.rel, self.kappa, self.floor)

    def to_dict(self) -> dict[str, Any]:
        return {'gain': self.gain, 'rel': self.rel, 'kappa': self.kappa,
                'floor': self.floor, 'info': dict(self.info)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> 'NoiseModel':
        """Make a model from `to_dict` output. Other keys are ignored."""
        return cls(gain=float(d['gain']), rel=float(d.get('rel', 0.0)),
                   kappa=float(d.get('kappa', 0.0)), floor=float(d.get('floor', 0.0)),
                   info=dict(d.get('info', {})))


# ──────────────────────────────────────────────────────────────────
# Gradient term
# ──────────────────────────────────────────────────────────────────

def _axis_slope2(f: npt.NDArray[np.float32], axis: int) -> npt.NDArray[np.float32]:
    """Square of the larger one-sided change per voxel along `axis`, NaN-aware.

    a = f[i+1] - f[i] and b = f[i] - f[i-1], in float32. The result is the
    larger one of a^2 and b^2 (float32). A square that is not finite (a
    missing neighbour) counts as 0. The edge voxels have only one of a and
    b.
    """
    n = f.shape[axis]
    out = np.zeros(f.shape, np.float32)
    if n < 2:
        return out
    with np.errstate(invalid='ignore', over='ignore'):
        d2 = np.diff(f, axis=axis)                    # d2[i] = f[i+1] - f[i]
        np.square(d2, out=d2)
    d2[~np.isfinite(d2)] = 0.0

    def sl(a: int, b: int) -> tuple[slice, ...]:
        s = [slice(None)] * 3
        s[axis] = slice(a, b)
        return tuple(s)

    np.maximum(d2[sl(1, n - 1)], d2[sl(0, n - 2)], out=out[sl(1, n - 1)])
    out[sl(0, 1)] = d2[sl(0, 1)]
    out[sl(n - 1, n)] = d2[sl(n - 2, n - 1)]
    return out


GRADIENT_BACKENDS = ('auto', 'numpy', 'numba', 'gpu')

_GRAD_CUDA = r'''
__device__ __forceinline__ bool fin(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) < 0x7f800000u;
}
__device__ __forceinline__ float sq0(float d) {
    float s = __fmul_rn(d, d);
    return fin(s) ? s : 0.0f;
}
// f: m h-layers (a slab with its neighbour layers), out: cnt layers from layer a
extern "C" __global__ void grad_kernel(const float* __restrict__ f, float* __restrict__ out,
                                       long long m, long long nk, long long nl,
                                       long long a, long long cnt)
{
    long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long nkl = nk * nl;
    if (t >= cnt * nkl) return;
    long long i = a + t / nkl;
    long long r = t - (t / nkl) * nkl;
    long long j = r / nl;
    long long k = r - j * nl;
    long long p = i * nkl + r;
    float x = f[p];
    if (!fin(x)) { out[t] = 0.0f; return; }
    float fa, fb;
    fa = i + 1 < m ? sq0(__fsub_rn(f[p + nkl], x)) : 0.0f;
    fb = i > 0 ? sq0(__fsub_rn(x, f[p - nkl])) : 0.0f;
    float s0 = fmaxf(fa, fb);
    fa = j + 1 < nk ? sq0(__fsub_rn(f[p + nl], x)) : 0.0f;
    fb = j > 0 ? sq0(__fsub_rn(x, f[p - nl])) : 0.0f;
    float s1 = fmaxf(fa, fb);
    fa = k + 1 < nl ? sq0(__fsub_rn(f[p + 1], x)) : 0.0f;
    fb = k > 0 ? sq0(__fsub_rn(x, f[p - 1])) : 0.0f;
    float s2 = fmaxf(fa, fb);
    out[t] = __fsqrt_rn(__fadd_rn(__fadd_rn(s0, s1), s2));
}
'''
_GRAD_KERNEL: list[Any] = []
_GRAD_SLAB_VOXELS = 1 << 26        # voxels of one GPU slab (256 MB in float32)


def _gradient_gpu(mean: npt.NDArray[np.floating]) -> npt.NDArray[np.float32]:
    """`gradient_magnitude` with a CUDA kernel, in slabs of h-layers."""
    import cupy as cp
    if not _GRAD_KERNEL:
        kern = cp.RawKernel(_GRAD_CUDA, 'grad_kernel', options=('--fmad=false',))
        kern.compile()
        _GRAD_KERNEL.append(kern)
    kern = _GRAD_KERNEL[0]
    nh, nk, nl = mean.shape
    nkl = nk * nl
    out = np.empty(mean.shape, np.float32)
    step = max(1, _GRAD_SLAB_VOXELS // max(nkl, 1))
    threads = 256
    try:
        for h0 in range(0, nh, step):
            h1 = min(nh, h0 + step)
            lo, hi = max(0, h0 - 1), min(nh, h1 + 1)
            d_f = cp.asarray(np.ascontiguousarray(mean[lo:hi], dtype=np.float32))
            cnt = h1 - h0
            d_o = cp.empty(cnt * nkl, cp.float32)
            kern(((cnt * nkl + threads - 1) // threads,), (threads,),
                 (d_f, d_o, np.int64(hi - lo), np.int64(nk), np.int64(nl),
                  np.int64(h0 - lo), np.int64(cnt)))
            d_o.get(out=out[h0:h1].reshape(-1))
            del d_f, d_o
    finally:
        oe._free_gpu_memory()
    return out


def gradient_magnitude(mean: npt.NDArray[np.floating], slab: int = 32,
                       backend: str = 'auto') -> npt.NDArray[np.float32]:
    """Gradient magnitude G (intensity change per voxel) of a volume.

    G = sqrt(d0^2 + d1^2 + d2^2), in float32. d_a^2 is the larger one of the
    squares of the two one-sided differences along array axis a
    (`_axis_slope2`). A missing neighbour (NaN) gives no difference. G is
    not 0 at the maximum of a peak (a minmod or a central difference is 0
    there). Thus the gradient guard of the noise model also covers the
    members at the peak maximum. G is 0 where `mean` is NaN.

    `backend` is 'numpy' (in slabs of `slab` h-layers, to limit the
    memory), 'numba', 'gpu' (CuPy) or 'auto' (numba if it is installed, else
    NumPy). All backends give the same bits. If the GPU backend fails, a
    warning is given and a CPU backend is used.
    """
    if backend not in GRADIENT_BACKENDS:
        raise ValueError(f"backend must be one of {GRADIENT_BACKENDS}, got {backend!r}")
    if mean.ndim != 3:
        raise ValueError(f"gradient_magnitude needs a 3D volume, got the shape {mean.shape}")
    if backend == 'gpu':
        try:
            return _gradient_gpu(mean)
        except Exception as exc:
            if not isinstance(exc, oe._gpu_errors()):
                raise
            import warnings
            warnings.warn(f"The GPU gradient failed ({exc!r}). The CPU calculates it "
                          "(slower, same result).", RuntimeWarning, stacklevel=2)
            backend = 'auto'
    if backend == 'numba' or (backend == 'auto' and oe.numba_available()):
        from . import _orbit_numba
        f = np.ascontiguousarray(mean, dtype=np.float32)
        out = np.empty(f.shape, np.float32)
        _orbit_numba.gradient_blocks(f, out)
        return out
    nh = mean.shape[0]
    out = np.empty(mean.shape, np.float32)
    for h0 in range(0, nh, slab):
        h1 = min(nh, h0 + slab)
        lo, hi = max(0, h0 - 1), min(nh, h1 + 1)
        f = np.asarray(mean[lo:hi], dtype=np.float32)
        g2 = _axis_slope2(f, 0)
        g2 += _axis_slope2(f, 1)
        g2 += _axis_slope2(f, 2)
        np.sqrt(g2, out=g2)
        g2[~np.isfinite(f)] = 0.0
        out[h0:h1] = g2[h0 - lo:h0 - lo + (h1 - h0)]
    return out


def _check_gradient(grad: Any, shape: tuple[int, ...]) -> npt.NDArray[np.float32]:
    """`grad` as a C-contiguous float32 array. Raises ValueError if its shape is not `shape`."""
    g = np.ascontiguousarray(grad, dtype=np.float32)
    if g.shape != tuple(shape):
        raise ValueError(f"grad has the shape {g.shape}, the volume has the shape {tuple(shape)}")
    return g


FIRST_PASS_SIGMA = 3.0


def first_pass_gradient(plan: oe.OrbitPlan, data: npt.NDArray[np.float32],
                        use_gpu: bool | None = None) -> npt.NDArray[np.float32]:
    """G (`gradient_magnitude`) of the first-pass symmetric average of `data`.

    The first pass has equal weights and the MAD rejection with sigma 3,
    min_valid 3 and no Poisson floor. A plain mean (no rejection) is not
    used: contaminated members move it, and G then becomes too large at many
    voxels. On synthetic data with 5 to 20 % contaminated members, G of the
    plain mean gave kappa 63 to 82 % too small in `fit_noise_model`. The
    Poisson floor is off because it makes the rejection too weak on a volume
    of pixel means (a rawrecon volume).

    If the GPU engine calculated the first pass, the GPU also calculates G.
    Else numba calculates G if it is installed, else NumPy.
    """
    if use_gpu is None:
        from .volume_builder import HAS_GPU
        use_gpu = HAS_GPU
    res = oe.run(plan, data, None, FIRST_PASS_SIGMA, 3, False, bool(use_gpu),
                 averaging=oe.Averaging(weighting='equal', rejection='mad'))
    mean = res.intensity
    backend = 'gpu' if res.engine == 'gpu' else 'auto'
    del res
    return gradient_magnitude(mean, backend=backend)


# ──────────────────────────────────────────────────────────────────
# Sample of orbits
# ──────────────────────────────────────────────────────────────────

@dataclass
class OrbitSample:
    """Orbit members of a sample of voxels (see `sample_orbits`)."""
    index: npt.NDArray[np.int64]                # (nv,) flat voxel index
    shell: npt.NDArray[np.int64]                # (nv,) |q| shell
    vals: npt.NDArray[np.float32]               # (n_ops, nv), NaN where missing
    ncnt: npt.NDArray[np.float64] | None        # (n_ops, nv) member pixel counts


def _quotas(available: npt.NDArray[np.int64], total: int) -> npt.NDArray[np.int64]:
    """Equal quota per shell. A shell with too few voxels gives its rest to the others."""
    quota = np.zeros(len(available), np.int64)
    left = int(total)
    open_ = [i for i in range(len(available)) if available[i] > 0]
    while left > 0 and open_:
        share = max(1, left // len(open_))
        nxt = []
        for i in open_:
            take = min(share, int(available[i] - quota[i]), left)
            quota[i] += take
            left -= take
            if quota[i] < available[i]:
                nxt.append(i)
            if left == 0:
                break
        open_ = nxt
    return quota


def sample_orbits(plan: oe.OrbitPlan, data: npt.NDArray[np.float32],
                  counts: npt.NDArray[np.float32] | None = None, n_members: int = 2_000_000,
                  seed: int = 0, min_members: int = 3) -> OrbitSample:
    """Sample of voxels, stratified by |q| shell, with their orbit members.

    The function takes random voxels with at least `min_members` finite
    members and no two members that read the same source voxel (general
    positions). It takes about the same number of voxels from each of the
    10 |q| shells (a shell with too few voxels gives its rest to the others),
    up to `n_members` / n_ops voxels in total.
    """
    rng = np.random.default_rng(seed)
    nvox = int(np.prod(plan.shape))
    target = max(1, int(n_members) // plan.n_ops)
    pool = min(nvox, 6 * target)
    if pool >= nvox:
        idx = np.arange(nvox, dtype=np.int64)
    else:
        idx = np.unique(rng.integers(0, nvox, pool, dtype=np.int64))
    src = oe.make_source(plan, data, counts=counts)
    keep_idx, keep_vals, keep_cnt = [], [], []
    step = max(1, 4_000_000 // plan.n_ops)
    nh, nk, nl = plan.shape
    for s in range(0, idx.size, step):
        part = idx[s:s + step]
        c = [a.astype(np.int64) for a in np.unravel_index(part, plan.shape)]
        dc = [a.astype(np.float64) for a in c]
        vals, _, ncnt = oe.gather_members(plan, src, c, dc, (part.size,))
        fin = np.isfinite(vals)
        # Source key of each finite member. A missing member gets a unique key.
        keys = np.empty(vals.shape, np.int64)
        for o, sd in enumerate(oe._sources(plan, None, c, dc)):
            s3 = [np.broadcast_to(sd[d], part.shape) for d in range(3)]
            one = (s3[0] >= 0) & (s3[1] >= 0) & (s3[2] >= 0)
            keys[o] = np.where(one, (s3[0] * nk + s3[1]) * nl + s3[2], -1 - o)
        keys = np.where(fin, keys, -1000 - np.arange(plan.n_ops)[:, None])
        ks = np.sort(keys, axis=0)
        dup = np.any(ks[1:] == ks[:-1], axis=0)
        ok = (fin.sum(axis=0) >= min_members) & ~dup
        keep_idx.append(part[ok])
        keep_vals.append(vals[:, ok])
        if ncnt is not None:
            keep_cnt.append(ncnt[:, ok])
    index = np.concatenate(keep_idx)
    vals = np.concatenate(keep_vals, axis=1)
    ncnt = np.concatenate(keep_cnt, axis=1) if keep_cnt else None
    c = np.unravel_index(index, plan.shape)
    shell = oe._shell_of(plan, *c)
    avail = np.bincount(shell, minlength=oe.N_SHELLS)
    quota = _quotas(avail, target)
    pick = []
    for sh in range(oe.N_SHELLS):
        where = np.flatnonzero(shell == sh)
        if quota[sh] < where.size:
            where = np.sort(rng.choice(where, int(quota[sh]), replace=False))
        pick.append(where)
    sel = np.sort(np.concatenate(pick))
    return OrbitSample(index[sel], shell[sel], vals[:, sel],
                       ncnt[:, sel] if ncnt is not None else None)


# ──────────────────────────────────────────────────────────────────
# Fit of the noise model
# ──────────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def median_variance_factor(m: int) -> float:
    """q_m = m Var(median of m independent N(0, 1) values).

    q_1 = q_2 = 1, and q_m goes to pi / 2 for a large m. A Monte Carlo
    calculation with a fixed seed gives it (about 1e6 values, relative error
    about 0.3 %).
    """
    if m < 1:
        raise ValueError("q_m needs m >= 1")
    if m <= 2:
        return 1.0
    rng = np.random.default_rng(20261010 + m)
    x = rng.standard_normal((max(1000, 1_000_000 // m), m))
    return float(m * np.var(np.median(x, axis=1)))


def _loo_median(x: npt.NDArray[np.float64], active: npt.NDArray[np.bool_]) -> tuple[Any, Any]:
    """Median of the other active members of each member (leave one out).

    Returns the median (n_ops, nv) and the number m of the other active
    members. The value is NaN where m = 0.
    """
    n_ops = x.shape[0]
    n = active.sum(axis=0)
    order = np.argsort(np.where(active, x, np.inf), axis=0, kind='stable')
    s = np.take_along_axis(np.where(active, x, np.inf), order, 0)
    rank = np.empty_like(order)
    np.put_along_axis(rank, order, np.arange(n_ops)[:, None].repeat(x.shape[1], 1), 0)
    m = np.where(active, n - 1, n)                  # other active members

    def at(j: Any) -> Any:
        # j-th smallest of the others: skip the own rank of an active member
        j = np.where(active & (j >= rank), j + 1, j)
        return np.take_along_axis(s, np.clip(j, 0, n_ops - 1), 0)

    lo = at(np.maximum((m - 1) // 2, 0))
    hi = at(m // 2)
    with np.errstate(invalid='ignore'):
        med = np.where(m > 0, 0.5 * (lo + hi), np.nan)
    return med, m


def _grouped_median(z: npt.NDArray[np.float64], group: npt.NDArray[np.int64],
                    count: npt.NDArray[np.int64]) -> npt.NDArray[np.float64]:
    """Median of z in each group (groups 0 .. len(count) - 1, count > 0)."""
    order = np.lexsort((z, group))
    zs = z[order]
    start = np.concatenate([[0], np.cumsum(count)[:-1]])
    return 0.5 * (zs[start + (count - 1) // 2] + zs[start + count // 2])


# Quantile levels of the bin edges of each model term. The extra levels at
# the top give sparse features (the flanks of Bragg peaks) their own bins.
FIT_NO_SCATTER = ("The noise-model fit cannot continue. The orbit members show no scatter "
                  "(all residuals are 0). The volume can be constant.")
FIT_NOT_FINITE = ("The noise-model fit cannot continue. It got values that are not finite. "
                  "Make sure that the measured voxels have finite intensities and pixel "
                  "counts above 0.")
FIT_NUMERIC = "The noise-model fit stopped because of a numerical problem."

_BIN_LEVELS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99, 0.995, 0.999)


def _binned_fit(z: npt.NDArray[np.float64], p: npt.NDArray[np.float64],
                theta0: npt.NDArray[np.float64] | None,
                min_bin: int) -> tuple[npt.NDArray[np.float64], int]:
    """Fit z = p @ theta (theta >= 0) from the medians of z in bins of p.

    `p` is (count, 4): max(med, 1) / N, med^2, G^2 and 1. The bins are
    quantile bins of the first three columns (`_BIN_LEVELS`). The fit is a
    non-negative least-squares fit of the bin medians with relative errors
    (weight sqrt(bin size) / model value). Bins with less than `min_bin`
    members are not used. Returns theta and the number of bins.
    """
    from scipy.optimize import nnls
    if not (np.all(np.isfinite(z)) and np.all(np.isfinite(p))):
        raise ValueError(FIT_NOT_FINITE)
    if not np.any(z > 0):
        raise ValueError(FIT_NO_SCATTER)
    code = np.zeros(z.size, np.int64)
    for j in range(3):
        edges = np.unique(np.quantile(p[:, j], _BIN_LEVELS))
        code = code * (len(_BIN_LEVELS) + 1) + np.searchsorted(edges, p[:, j], side='right')
    _, group, count = np.unique(code, return_inverse=True, return_counts=True)
    medz = _grouped_median(z, group, count)
    X = np.stack([np.bincount(group, weights=p[:, j]) / count for j in range(4)], axis=1)
    sel = count >= min_bin
    if sel.sum() < 4:
        sel = count > 0
    X, y, c = X[sel], medz[sel], count[sel].astype(np.float64)
    scale = np.max(np.abs(X), axis=0)
    scale[scale == 0] = 1.0
    Xs = X / scale
    pred = X @ theta0 if theta0 is not None else y.copy()
    tiny = max(1e-12 * float(np.max(np.abs(y))), 1e-300)
    theta = np.zeros(4)
    for _ in range(4):
        wrow = np.sqrt(c) / np.maximum(pred, tiny)
        try:
            sol, _ = nnls(Xs * wrow[:, None], y * wrow)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as exc:
            raise ValueError(f"{FIT_NUMERIC} ({type(exc).__name__}: {exc})") from exc
        theta = sol / scale
        pred = X @ theta
    return theta, int(sel.sum())


def _predictors(med: npt.NDArray[np.float64], ncnt: Any,
                g: npt.NDArray[np.float64]) -> list[Any]:
    """Columns of the linear model: max(med, 1) / N, med^2, G^2, 1."""
    return [np.maximum(med, 1.0) / ncnt, med * med, g * g, 1.0]


def fit_members(vals: npt.NDArray[np.floating], ncnt: npt.NDArray[np.float64] | None,
                grad: npt.NDArray[np.floating], seed: int = 0, clip: float = 4.0,
                max_iter: int = 10, min_bin: int = 30) -> NoiseModel:
    """Fit the noise model to orbit members. See `fit_noise_model`.

    `vals` is (n_ops, nv) with NaN where a member is missing, `ncnt` the
    member pixel counts (or None for N = 1) and `grad` G at the nv target
    voxels.
    """
    rng = np.random.default_rng(seed)
    x = np.asarray(vals, np.float64)
    if x.ndim != 2 or x.shape[1] == 0:
        raise ValueError(NO_ORBITS)
    fin = np.isfinite(x)
    n_ops = x.shape[0]
    finite_values = x[fin]
    dither = bool(finite_values.size) and bool(np.all(finite_values == np.round(finite_values)))
    del finite_values
    if dither:
        # integer data: a uniform dither of +-0.5 makes the residuals continuous
        x = x + rng.uniform(-0.5, 0.5, x.shape)
    ncnt_a: Any = np.asarray(ncnt, np.float64) if ncnt is not None else 1.0
    g = np.asarray(grad, np.float64)
    qm = np.array([1.0] + [median_variance_factor(m) for m in range(1, n_ops + 1)])
    active = fin.copy()
    theta = None
    it = 0
    n_bins_used = 0

    def model(th: npt.NDArray[np.float64], cols: list[Any]) -> npt.NDArray[np.float64]:
        v = sum(th[j] * np.broadcast_to(cols[j], x.shape) for j in range(4))
        # a fit with some zero terms can give v = 0: keep v positive
        vpos = v[active & (v > 0)]
        return np.maximum(v, 1e-6 * float(np.median(vpos)) if vpos.size else 1e-300)

    def median_var(v: npt.NDArray[np.float64], m: npt.NDArray[np.int64]) -> Any:
        # V_i = q_m / sum_j (1 / v_j) over the other active members j
        inv = np.where(active, 1.0 / v, 0.0)
        return qm[m] / (inv.sum(axis=0) - inv)

    with np.errstate(invalid='ignore', divide='ignore'):
        for it in range(1, max_iter + 1):
            n = active.sum(axis=0)
            vox = n >= 3
            # orbit median of the active members (the model uses it)
            srt = np.sort(np.where(active, x, np.inf), axis=0)
            lo = np.take_along_axis(srt, np.maximum((n - 1) // 2, 0)[None], 0)[0]
            hi = np.take_along_axis(srt, np.minimum(n // 2, n_ops - 1)[None], 0)[0]
            med = np.where(vox, 0.5 * (lo + hi), np.nan)
            del srt
            cols = _predictors(med, ncnt_a, g)
            # residual about the median of the other active members
            loo, m = _loo_median(x, active)
            r2 = (x - loo) ** 2
            if theta is None:
                shrink = m / (m + qm[m])           # equal variances: v / (v + q_m v / m)
            else:
                v = model(theta, cols)
                shrink = v / (v + median_var(v, m))
            # the median of z is v_i: r^2 / (v_i + V_i) is about chi^2(1)
            z = shrink * r2 / CHI2_MEDIAN
            use = active & vox
            p = np.stack([np.broadcast_to(cc, x.shape)[use] for cc in cols], axis=1)
            new_theta, n_bins_used = _binned_fit(z[use], p, theta, min_bin)
            del p
            v = model(new_theta, cols)
            std2 = r2 / (v + median_var(v, m))
            # a voxel with less than 3 active members keeps its flags
            new_active = np.where(vox, fin & ~(std2 > clip * clip), active)
            done = (theta is not None
                    and np.allclose(new_theta, theta, rtol=1e-4, atol=1e-12 * np.max(new_theta))
                    and np.array_equal(new_active, active))
            theta = new_theta
            active = new_active
            if done:
                break
    floor2 = theta[3] - (1.0 / 12.0 if dither else 0.0)
    info = {'method': 'binned medians of squared leave-one-out residuals, '
                      'clipped and iterated',
            'n_voxels': int(x.shape[1]), 'n_members': int(fin.sum()),
            'n_bins': n_bins_used, 'iterations': it, 'clip': float(clip),
            'flagged_fraction': float(1.0 - active[fin].mean()) if fin.any() else 0.0,
            'dithered': dither}
    gain = float(max(theta[0], 0.0))
    rel = float(np.sqrt(max(theta[1], 0.0)))
    kappa = float(np.sqrt(max(theta[2], 0.0)))
    flr = float(np.sqrt(max(floor2, 0.0)))
    info['floor_added'] = False
    if gain <= 0 and flr <= 0:
        # Only rel and kappa: v = 0 at med = 0 and G = 0. A small floor keeps v > 0.
        th = np.array([0.0, theta[1], theta[2], 0.0])
        with np.errstate(invalid='ignore'):
            v = sum(th[j] * np.broadcast_to(cols[j], x.shape) for j in range(4))
        vpos = v[active & np.isfinite(v) & (v > 0)]
        if (rel <= 0 and kappa <= 0) or vpos.size == 0:
            raise ValueError("The noise-model fit gives gain = 0, rel = 0, kappa = 0 and "
                             "floor = 0. The data have no measurable scatter between orbit "
                             "members.")
        flr = float(np.sqrt(FLOOR_ADDED * float(np.median(vpos))))
        info['floor_added'] = True
    return NoiseModel(gain=gain, rel=rel, kappa=kappa, floor=flr, info=info)


def _fit(vol: Any, laue_group: str, plan: oe.OrbitPlan | None = None,
         data: npt.NDArray[np.float32] | None = None, use_gpu: bool | None = None,
         n_members: int = 2_000_000, seed: int = 0,
         grad: npt.NDArray[np.float32] | None = None,
         **kwargs: Any) -> tuple[NoiseModel, npt.NDArray[np.float32]]:
    """`fit_noise_model` that also returns the gradient volume G.

    If `grad` is not None, the fit uses it and does not calculate G again.
    """
    from .volume_builder import orbit_plan
    t0 = time.perf_counter()
    if plan is None:
        plan = orbit_plan(vol, laue_group)
    if data is None:
        data = np.ascontiguousarray(vol.intensity, dtype=np.float32)
    if grad is None:
        grad = first_pass_gradient(plan, data, use_gpu)
    else:
        grad = _check_gradient(grad, data.shape)
    t1 = time.perf_counter()
    counts = (np.ascontiguousarray(vol.counts, dtype=np.float32)
              if vol.counts is not None else None)
    smp = sample_orbits(plan, data, counts, n_members=n_members, seed=seed)
    del counts
    model = fit_members(smp.vals, smp.ncnt, grad.reshape(-1)[smp.index], seed=seed, **kwargs)
    model.info.update({'laue_group': laue_group, 'seed': int(seed),
                       'has_counts': vol.counts is not None,
                       't_first_pass_s': t1 - t0, 't_fit_s': time.perf_counter() - t1})
    return model, grad


def fit_noise_model(vol: Any, laue_group: str, n_members: int = 2_000_000, seed: int = 0,
                    use_gpu: bool | None = None, clip: float = 4.0,
                    max_iter: int = 10, return_gradient: bool = False
                    ) -> NoiseModel | tuple[NoiseModel, npt.NDArray[np.float32]]:
    """Fit the noise model (module docstring) to the orbits of a volume.

    Returns the model. With `return_gradient=True`, it returns (model, G).
    G is the gradient volume of step 1. Give G to `symmetrize_volume` and
    `score_candidates` (argument `grad`), so that they do not calculate it
    again.

    Method:

    1. First pass: the symmetric average of the volume with equal weights
       and the MAD rejection (`first_pass_gradient`), on the GPU if
       `use_gpu`. G is its gradient magnitude (`gradient_magnitude`).
    2. Sample: up to `n_members` orbit members of random voxels, about the
       same number from each |q| shell (`sample_orbits`). Each voxel has at
       least 3 finite members, all from different source voxels.
    3. If all finite values are integers (raw counts), a uniform dither of
       +-0.5 is added to each value. This makes the residuals continuous.
       The dither variance 1/12 is subtracted from floor^2 at the end.
    4. Residual of each member: r_i = x_i - med_-i. med_-i is the median of
       the m other active members of the orbit, thus r_i does not depend on
       x_i. Its variance is v_i + V_i. V_i = q_m / sum_j (1 / v_j) is the
       variance of med_-i (j: the other active members, q_m: the factor of
       the finite orbit size, `median_variance_factor`). Thus
       r_i^2 / (v_i + V_i) is chi^2(1), and
       z_i = r_i^2 v_i / ((v_i + V_i) 0.4549) has the median v_i
       (0.4549 is the median of chi^2(1)). The first iteration uses
       V_i = q_m v_i / m. Later iterations use v from the last fit.
    5. The members are put in bins of the three model terms
       max(med, 1) / N, med^2 and G^2 (med: the median of the active
       members). The bin edges are quantiles (10 % steps and 0.95, 0.98,
       0.99, 0.995, 0.999). A non-negative least-squares fit of the bin
       medians of z, with relative errors, gives gain, rel^2, kappa^2 and
       floor^2. Medians, not means, are used, so that contaminated members
       do not increase the fit.
    6. A member with r_i^2 / (v_i + V_i) > `clip`^2 is flagged (made not
       active). Steps 4 and 5 are done again until the parameters and the
       flags do not change, or `max_iter` times. This removes the effect of
       contamination on the medians.
    7. If the fit gives gain = 0 and floor = 0, but rel > 0 or kappa > 0,
       floor^2 is set to `FLOOR_ADDED` times the median model variance of
       the sample. This keeps v > 0 at med = 0 and G = 0.
       `info['floor_added']` is then True.

    The fit assumes Gaussian residuals. At low counts (a few counts per
    member) Poisson residuals are skewed, and the fit is an approximation.
    `NoiseModel.info` holds the details of the fit.

    The function raises ValueError if the sample has no orbit with 3 or
    more members from different sources (`NO_ORBITS`), or if all four
    parameters are 0.
    """
    model, grad = _fit(vol, laue_group, use_gpu=use_gpu, n_members=n_members, seed=seed,
                       clip=clip, max_iter=max_iter)
    return (model, grad) if return_gradient else model


# ──────────────────────────────────────────────────────────────────
# Scores of averaging options
# ──────────────────────────────────────────────────────────────────

EQUAL_MAD = {'name': 'equal, mad 3', 'weighting': 'equal', 'rejection': 'mad', 'sigma': 3.0}
COUNTS_MAD = {'name': 'counts, mad 3', 'weighting': 'counts', 'rejection': 'mad', 'sigma': 3.0}
ADAPTIVE_MODEL = {'name': 'adaptive, model 4', 'weighting': 'adaptive', 'rejection': 'model',
                  'sigma': 4.0}
# The reference of `bias_checks` for `suggest`: the default of rspace3d 2.3.0
BIAS_REFERENCE = EQUAL_MAD['name']


def automatic_candidates(has_counts: bool) -> list[dict[str, Any]]:
    """The candidates that the app compares automatically (`score_candidates`, `suggest`).

    'equal, mad 3' (the default of rspace3d 2.3.0 and the bias reference),
    'counts, mad 3' (only if the volume has pixel counts) and
    'adaptive, model 4'. Each candidate rejects outliers. An option without
    rejection is never in this set, because spikes then go into the mean.
    A user can still give any option to `symmetrize_volume`.
    """
    cands = [dict(EQUAL_MAD)]
    if has_counts:
        cands.append(dict(COUNTS_MAD))
    cands.append(dict(ADAPTIVE_MODEL))
    return cands


def _candidate(spec: Mapping[str, Any]) -> tuple[str, oe.Averaging, float | None, int, bool]:
    """Name, engine options, sigma, min_valid and poisson_floor of a candidate."""
    weighting = spec.get('weighting', 'equal')
    rejection = spec.get('rejection', 'none')
    sigma = spec.get('sigma', 3.0 if rejection != 'none' else None)
    sigma = None if rejection == 'none' else float(sigma)
    name = spec.get('name', f"{weighting}, {rejection}" + (f" {sigma:g}" if sigma else ''))
    min_valid = int(np.ceil(float(spec.get('min_valid', 3))))
    return (str(name), oe.Averaging(weighting=weighting, rejection=rejection), sigma,
            min_valid, bool(spec.get('poisson_floor', True)))


def _stats(z: npt.NDArray[np.float64], trim: float) -> dict[str, Any]:
    n = int(z.size)
    if n == 0:
        return {'trimmed_mean': None, 'mean': None, 'n_members': 0}
    keep = n - int(np.ceil(trim * n))
    zs = np.sort(z)[:max(keep, 1)]
    return {'trimmed_mean': float(zs.mean()), 'mean': float(z.mean()), 'n_members': n}


def score_candidates(vol: Any, laue_group: str,
                     candidates: Sequence[Mapping[str, Any]] | None = None,
                     noise_model: NoiseModel | None = None, n_members: int = 1_000_000,
                     seed: int = 1, use_gpu: bool | None = None, trim: float = 0.01,
                     progress_callback: Callable[[int, int], None] | None = None,
                     grad: npt.NDArray[np.float32] | None = None) -> dict[str, Any]:
    """Leave-one-member-out score of each averaging option. Lower is better.

    `candidates` is a list of dicts with the keys 'name', 'weighting',
    'rejection', 'sigma', 'min_valid' and 'poisson_floor' (the keywords of
    `symmetrize_volume`). None gives `automatic_candidates`. For each
    sampled voxel (`sample_orbits`) and each finite member i, the candidate
    estimator calculates the voxel from the other members (est_-i). The
    estimator is `orbit_engine.estimate`, the code of the NumPy engine. The
    score of the member is

        z_i = (x_i - est_-i)^2 / v_i,

    with v_i from `noise_model` at the median of all finite members. All
    candidates use the same noise model, the same voxels and the same v_i.
    If `noise_model` is None, `fit_noise_model` gives it (a failed fit
    raises its ValueError). `grad` is the gradient volume G of
    `fit_noise_model(..., return_gradient=True)`. If it is None and the
    model has kappa > 0, the function calculates G.

    All candidates are scored on the same members: the sampled members for
    which all candidates give a finite estimate. A candidate can give no
    estimate for a member (for example if its rejection removes all other
    members). 'n_no_estimate' counts these members for each candidate.
    `suggest` does not suggest a candidate with n_no_estimate > 0.

    The sample has only general positions: voxels where no two members read
    the same source voxel (`sample_orbits`). Thus the voxels on mirror
    planes and rotation axes are not scored.

    Returns a dict with 'candidates' (name -> {'spec', 'n_no_estimate',
    'shells', 'total'}), 'has_counts', 'n_members_sampled' and
    'n_members_scored'. 'shells' holds lists per |q| shell and 'total' the
    values of all scored members: 'trimmed_mean' (mean of z without the
    largest `trim` fraction), 'mean' and 'n_members'.
    """
    from .volume_builder import orbit_plan
    plan = orbit_plan(vol, laue_group)
    data = np.ascontiguousarray(vol.intensity, dtype=np.float32)
    has_counts = vol.counts is not None
    if candidates is None:
        candidates = automatic_candidates(has_counts)
    if grad is not None:
        grad = _check_gradient(grad, data.shape)
    if noise_model is None:
        noise_model, grad = _fit(vol, laue_group, plan, data, use_gpu=use_gpu, seed=seed,
                                 grad=grad)
    elif noise_model.kappa > 0 and grad is None:
        grad = first_pass_gradient(plan, data, use_gpu)
    counts = (np.ascontiguousarray(vol.counts, dtype=np.float32) if has_counts else None)
    smp = sample_orbits(plan, data, counts, n_members=n_members, seed=seed)
    del counts
    g = (grad.reshape(-1)[smp.index] if grad is not None and noise_model.kappa > 0
         else np.zeros(smp.index.size, np.float32))
    del grad
    n_ops, nv = smp.vals.shape
    fin = np.isfinite(smp.vals)
    with np.errstate(invalid='ignore'):
        med = oe._row_median(smp.vals, fin.sum(axis=0))
    v_ref = np.broadcast_to(oe.model_variance(med, smp.ncnt, g, noise_model.params),
                            smp.vals.shape)

    specs = [_candidate(c) for c in candidates]
    if len({s[0] for s in specs}) != len(specs):
        raise ValueError("the candidate names must be unique")
    z_all: dict[str, list[Any]] = {s[0]: [] for s in specs}
    e_all: dict[str, list[Any]] = {s[0]: [] for s in specs}
    sh_all: list[Any] = []
    chunk = max(1, 4_000_000 // (n_ops * n_ops))
    steps = len(range(0, nv, chunk)) * len(specs)
    step = 0
    for s in range(0, nv, chunk):
        sl = slice(s, min(nv, s + chunk))
        x = smp.vals[:, sl]
        cv = x.shape[1]
        # column (i, voxel): the orbit without member i
        loo = np.repeat(x[None], n_ops, axis=0)            # (i, op, voxel)
        loo[np.arange(n_ops), np.arange(n_ops)] = np.nan
        loo = loo.transpose(1, 0, 2).reshape(n_ops, n_ops * cv)
        nc = (np.tile(smp.ncnt[:, sl], (1, n_ops)) if smp.ncnt is not None else None)
        gt = np.tile(g[sl], n_ops)
        ok = fin[:, sl]
        sh_all.append(np.broadcast_to(smp.shell[sl], (n_ops, cv))[ok])
        for name, av, sigma, min_valid, pfloor in specs:
            if progress_callback:
                progress_callback(step, steps)
            step += 1
            av.noise = noise_model.params
            est = oe.estimate(loo, nc, gt, sigma, min_valid, pfloor, av).out
            est = est.reshape(n_ops, cv).astype(np.float64)
            with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
                z = (x.astype(np.float64) - est) ** 2 / v_ref[:, sl]
            has_est = np.isfinite(est)
            z_all[name].append(np.where(has_est, z, np.nan)[ok])
            e_all[name].append(has_est[ok])
    if progress_callback:
        progress_callback(steps, steps)
    shells = np.concatenate(sh_all) if sh_all else np.zeros(0, np.int64)
    zs = {name: np.concatenate(z_all[name]) if z_all[name] else np.zeros(0) for name in z_all}
    # the common members: all candidates give a finite score
    common = np.ones(shells.size, bool)
    for z in zs.values():
        common &= np.isfinite(z)
    out: dict[str, Any] = {}
    for name, av, sigma, min_valid, pfloor in specs:
        z = zs[name]
        has_est = np.concatenate(e_all[name]) if e_all[name] else np.zeros(0, bool)
        per = [_stats(z[common & (shells == k)], trim) for k in range(oe.N_SHELLS)]
        spec = {'weighting': av.weighting, 'rejection': av.rejection, 'sigma': sigma,
                'min_valid': min_valid, 'poisson_floor': pfloor}
        out[name] = {'spec': spec, 'n_no_estimate': int((~has_est).sum()),
                     'shells': {key: [p[key] for p in per]
                                for key in ('trimmed_mean', 'mean', 'n_members')},
                     'total': _stats(z[common], trim)}
    return {'q_unit': plan.q_unit,
            'shell_edges': [plan.q_max * i / oe.N_SHELLS for i in range(oe.N_SHELLS + 1)],
            'n_voxels': int(nv), 'n_members_sampled': int(shells.size),
            'n_members_scored': int(common.sum()), 'has_counts': bool(has_counts),
            'trim': float(trim), 'noise_model': noise_model.to_dict(), 'candidates': out}


# ──────────────────────────────────────────────────────────────────
# Bias checks and suggestion
# ──────────────────────────────────────────────────────────────────

def _offset_label(o: Sequence[float]) -> str:
    return '(' + ', '.join(f"{float(v):g}" for v in o) + ')'


def _numba_region_sums(x: npt.NDArray[Any], ref: npt.NDArray[Any],
                       origin: npt.NDArray[np.float64], A: npt.NDArray[np.float64],
                       offsets: npt.NDArray[np.float64], radius: float,
                       plan: oe.OrbitPlan | None = None) -> tuple[Any, Any, Any]:
    """Sums of x and ref per |q| shell and region with the numba kernel.

    Returns (sums, reference sums, lost voxels), each (n_shells, regions).
    Without `plan`, n_shells = 1. With `plan`, the shells of the orbit engine.
    """
    from . import _orbit_numba
    x = np.ascontiguousarray(x)
    ref = np.ascontiguousarray(ref)
    if plan is None:
        G, scale, n_sh = np.eye(3), 0.0, 1
    else:
        G, scale, n_sh = plan.geo[12:21].reshape(3, 3).copy(), float(plan.geo[21]), oe.N_SHELLS
    nh = x.shape[0]
    n_reg = len(offsets) + 1
    s = np.zeros((nh, n_sh, n_reg))
    r = np.zeros((nh, n_sh, n_reg))
    n = np.zeros((nh, n_sh, n_reg), np.int64)
    _orbit_numba.region_sums(x, ref, np.ascontiguousarray(origin, np.float64),
                             np.ascontiguousarray(A, np.float64),
                             np.ascontiguousarray(offsets, np.float64).reshape(-1, 3),
                             float(radius) ** 2, G, scale, n_sh, s, r, n)
    # add the h-layers in a fixed order: the result does not depend on the threads
    out_s = np.zeros((n_sh, n_reg))
    out_r = np.zeros((n_sh, n_reg))
    for i in range(nh):
        out_s += s[i]
        out_r += r[i]
    return out_s, out_r, n.sum(axis=0)


def node_sums_by_shell(result: Any, reference: Any, laue_group: str,
                       radius: float = 0.1) -> dict[str, Any]:
    """Node sums of `result` and `reference` per |q| shell of the orbit engine.

    The node region is the same as in `bias_checks`: the distance in hkl to
    the nearest integer node is at most `radius`. Only voxels that are finite
    in both volumes are added. Returns {'result', 'reference', 'ratio'}
    (lists per shell, ratio None if the reference sum is 0).
    """
    from .volume_builder import orbit_plan, volume_affine
    plan = orbit_plan(reference, laue_group)
    origin, A = volume_affine(reference)
    x = np.asarray(getattr(result, 'intensity', result))
    ref = np.asarray(reference.intensity)
    if oe.numba_available():
        s, r, _ = _numba_region_sums(x, ref, origin, A, np.zeros((1, 3)), radius, plan)
        s, r = s[:, 0], r[:, 0]
    else:
        s = np.zeros(oe.N_SHELLS)
        r = np.zeros(oe.N_SHELLS)
        nh, nk, nl = ref.shape
        r2 = float(radius) ** 2
        jk = np.arange(nk, dtype=np.float64)[:, None]
        jl = np.arange(nl, dtype=np.float64)[None, :]
        for h0 in range(0, nh, 16):
            h1 = min(nh, h0 + 16)
            ih = np.arange(h0, h1, dtype=np.float64)[:, None, None]
            d2 = 0.0
            for d in range(3):
                f = origin[d] + A[d, 0] * ih + A[d, 1] * jk[None] + A[d, 2] * jl[None]
                f = f - np.rint(f)
                d2 = d2 + f * f
            c = np.indices((h1 - h0, nk, nl))
            sh = oe._shell_of(plan, c[0] + h0, c[1], c[2])
            xs = x[h0:h1].astype(np.float64)
            rs = ref[h0:h1].astype(np.float64)
            m = np.broadcast_to(d2 <= r2, xs.shape) & np.isfinite(xs) & np.isfinite(rs)
            s += np.bincount(sh[m], weights=xs[m], minlength=oe.N_SHELLS)
            r += np.bincount(sh[m], weights=rs[m], minlength=oe.N_SHELLS)
    return {'result': [float(v) for v in s], 'reference': [float(v) for v in r],
            'ratio': [float(a / b) if b != 0 else None for a, b in zip(s, r)]}


def bias_checks(results: Mapping[str, Any], reference: Any, radius: float = 0.1,
                extra_positions: Iterable[Sequence[float]] = (),
                slab: int = 16) -> dict[str, Any]:
    """Sums near Bragg nodes, near extra positions and in the diffuse region.

    `results` maps a name to a symmetrised volume (`VolumeData` or array of
    the same shape as `reference`). `reference` is a `VolumeData` (it also
    gives the grid). For `suggest`, use the 'equal, mad 3' result
    (`BIAS_REFERENCE`, the default of rspace3d 2.3.0). For each result, the
    function adds the voxels that are finite in the result and in the
    reference:

    * node: the distance in hkl (r.l.u.) to the nearest integer node is
      at most `radius`,
    * each extra position o (for example (0.5, 0.5, 0.5)): the distance to
      the nearest point n + o (n integer) is at most `radius`,
    * diffuse: all other voxels.

    The sums do not see a voxel that is NaN in the result. Thus the function
    also counts, in each region, the voxels that are finite in the
    reference and NaN in the result (lost voxels).

    If numba is installed, one parallel pass per result calculates the
    sums. The regions and the lost-voxel counts are the same as those of
    the NumPy code. The sums are float64 sums in a different order. Thus
    the numba sums and the NumPy sums can differ in the last bits (relative
    difference about 1e-15).

    Returns name -> {'node_sum_ratio', 'extra_sum_ratio' (label -> ratio),
    'diffuse_sum_ratio', 'sums', 'reference_sums', 'lost_voxels' (region
    -> count), 'n_lost' (sum of the counts)}. Each ratio is the sum of the
    result divided by the sum of the reference over the same voxels (None
    if the reference sum is 0). The regions are 'node', the labels of the
    extra positions and 'diffuse'.
    """
    from .volume_builder import volume_affine
    origin, A = volume_affine(reference)
    ref = np.asarray(reference.intensity)
    extra = [np.asarray(o, np.float64) for o in extra_positions]
    labels = [_offset_label(o) for o in extra]
    arrays = {name: np.asarray(getattr(r, 'intensity', r)) for name, r in results.items()}
    for name, a in arrays.items():
        if a.shape != ref.shape:
            raise ValueError(f"result {name!r} has the shape {a.shape}, not {ref.shape}")
    regions = ['node'] + labels + ['diffuse']
    sums = {name: dict.fromkeys(regions, 0.0) for name in arrays}
    rsums = {name: dict.fromkeys(regions, 0.0) for name in arrays}
    lost = {name: dict.fromkeys(regions, 0) for name in arrays}
    if oe.numba_available():
        offsets = np.array([np.zeros(3)] + extra, np.float64).reshape(-1, 3)
        for name, a in arrays.items():
            s, r, n = _numba_region_sums(a, ref, origin, A, offsets, radius)
            for j, reg in enumerate(regions):
                sums[name][reg] = float(s[0, j])
                rsums[name][reg] = float(r[0, j])
                lost[name][reg] = int(n[0, j])
        arrays = {}                      # the NumPy loop below has nothing to do
    nh, nk, nl = ref.shape
    jk = np.arange(nk, dtype=np.float64)[:, None]
    jl = np.arange(nl, dtype=np.float64)[None, :]
    r2 = float(radius) ** 2
    for h0 in range(0, nh if arrays else 0, slab):
        h1 = min(nh, h0 + slab)
        ih = np.arange(h0, h1, dtype=np.float64)[:, None, None]
        hkl = [origin[d] + A[d, 0] * ih + A[d, 1] * jk[None] + A[d, 2] * jl[None]
               for d in range(3)]

        def near(o: npt.NDArray[np.float64]) -> npt.NDArray[np.bool_]:
            d2 = 0.0
            for d in range(3):
                f = hkl[d] - o[d]
                f = f - np.rint(f)
                d2 = d2 + f * f
            return np.broadcast_to(d2 <= r2, (h1 - h0, nk, nl))

        masks = {'node': near(np.zeros(3))}
        for lab, o in zip(labels, extra):
            masks[lab] = near(o)
        inside = np.zeros((h1 - h0, nk, nl), bool)
        for m in masks.values():
            inside |= m
        masks['diffuse'] = ~inside
        rs = ref[h0:h1].astype(np.float64)
        rfin = np.isfinite(rs)
        for name, a in arrays.items():
            xs = a[h0:h1].astype(np.float64)
            xfin = np.isfinite(xs)
            both = rfin & xfin
            gone = rfin & ~xfin
            for reg in regions:
                m = masks[reg] & both
                sums[name][reg] += float(xs[m].sum())
                rsums[name][reg] += float(rs[m].sum())
                lost[name][reg] += int(np.count_nonzero(masks[reg] & gone))

    def ratio(a: float, b: float) -> float | None:
        return a / b if b != 0 else None

    out: dict[str, Any] = {}
    for name in sums:
        s, r = sums[name], rsums[name]
        out[name] = {'node_sum_ratio': ratio(s['node'], r['node']),
                     'extra_sum_ratio': {lab: ratio(s[lab], r[lab]) for lab in labels},
                     'diffuse_sum_ratio': ratio(s['diffuse'], r['diffuse']),
                     'sums': s, 'reference_sums': r, 'lost_voxels': lost[name],
                     'n_lost': int(sum(lost[name].values()))}
    return out


def preselect(has_counts: bool, cause: str = '') -> tuple[str, str]:
    """The safe preselection: 'counts, mad 3' with pixel counts, else 'equal, mad 3'.

    `suggest` returns it if no candidate passes the checks. Use it also if
    `fit_noise_model` raises ValueError (the app then has no scores).
    `cause` is a text that tells why there is no suggestion. Returns the
    name and a reason text.
    """
    if has_counts:
        name = COUNTS_MAD['name']
        why = ("The volume has pixel counts. Thus the preselection is the counts weights "
               "with the MAD rejection (sigma 3).")
    else:
        name = EQUAL_MAD['name']
        why = ("The volume has no pixel counts. Thus the preselection is the default of "
               "rspace3d 2.3.0: equal weights with the MAD rejection (sigma 3).")
    text = f"Use '{name}'. " + (cause.strip() + ' ' if cause.strip() else '') + why
    return name, text


def suggest(scores: Mapping[str, Any], bias: Mapping[str, Any], node_tol: float = 0.02,
            diffuse_tol: float = 0.02) -> tuple[str, str]:
    """The candidate with the lowest total trimmed score that passes all checks.

    `scores` is the output of `score_candidates` and `bias` the output of
    `bias_checks` with the 'equal, mad 3' result as the reference
    (`BIAS_REFERENCE`), with the same names. A candidate fails if:

    * it gives no estimate for some sampled members (n_no_estimate > 0),
    * it has no bias check,
    * it makes voxels NaN that are finite in the reference (n_lost > 0),
    * its node sum ratio, or the sum ratio of an extra position, differs
      from 1 by more than `node_tol`,
    * its diffuse sum ratio differs from 1 by more than `diffuse_tol`.

    If no candidate passes, the function returns the preselection
    (`preselect`: 'counts, mad 3' if the volume has pixel counts, else
    'equal, mad 3'). Returns the name and a reason text.
    """
    cands = scores['candidates']
    ranked = sorted((c for c in cands if cands[c]['total']['trimmed_mean'] is not None),
                    key=lambda c: cands[c]['total']['trimmed_mean'])
    notes = [f"'{c}' has no score." for c in cands if c not in ranked]
    for name in ranked:
        score = cands[name]['total']['trimmed_mean']
        fails = []
        n_no = int(cands[name].get('n_no_estimate', 0))
        if n_no > 0:
            fails.append(f"it gives no estimate for {n_no} sampled members")
        b = bias.get(name) if bias is not None else None
        if b is None:
            fails.append("it has no bias check")
        else:
            n_lost = int(b.get('n_lost', 0))
            if n_lost > 0:
                fails.append(f"it makes {n_lost} voxels NaN that are finite in the reference")
            checks = [('node sum ratio', b.get('node_sum_ratio'), node_tol)]
            checks += [(f"sum ratio at {lab}", val, node_tol)
                       for lab, val in b.get('extra_sum_ratio', {}).items()]
            checks.append(('diffuse sum ratio', b.get('diffuse_sum_ratio'), diffuse_tol))
            for what, val, tol in checks:
                if val is not None and abs(val - 1.0) > tol:
                    fails.append(f"its {what} is {val:.4f}")
        if not fails:
            text = [f"Use '{name}'. It has the lowest score ({score:.4g}) of the "
                    "candidates that pass the checks."]
            text += notes
            return name, ' '.join(text)
        notes.append(f"'{name}' (score {score:.4g}) fails because "
                     + ' and '.join(fails) + '.')
    name, text = preselect(bool(scores.get('has_counts', False)),
                           "No candidate passes the checks (node tolerance "
                           f"{node_tol:g}, diffuse tolerance {diffuse_tol:g}).")
    return name, ' '.join([text] + notes)


# ──────────────────────────────────────────────────────────────────
# Workflow: measure, compare, apply (averaging panel and command line)
# ──────────────────────────────────────────────────────────────────

LIMITS_TEXT = ("The score cannot see contamination that is the same in many members of an "
               "orbit (for example a powder ring). Look at the slices before you apply a choice.")


def make_candidate(weighting: str, rejection: str, sigma: float | None = None) -> dict[str, Any]:
    """A candidate dict with the standard name, for example 'counts, mad 3'."""
    if rejection == 'none':
        return {'name': f"{weighting}, none", 'weighting': weighting, 'rejection': 'none'}
    sigma = 3.0 if sigma is None else float(sigma)
    return {'name': f"{weighting}, {rejection} {sigma:g}", 'weighting': weighting,
            'rejection': rejection, 'sigma': sigma}


def more_candidates(has_counts: bool) -> list[dict[str, Any]]:
    """Options that a user can add by hand to the automatic set.

    The options without rejection are in this list for a comparison only.
    The panel and the command line never suggest them.
    """
    out = [make_candidate('equal', 'none')]
    if has_counts:
        out.append(make_candidate('counts', 'none'))
    out.append(make_candidate('adaptive', 'none'))
    if has_counts:
        out.append(make_candidate('counts', 'model', 4.0))
    out.append(make_candidate('equal', 'model', 4.0))
    return out


def known_candidate(name: str) -> dict[str, Any] | None:
    """The candidate dict of a standard name ('weighting, rejection [sigma]'), or None."""
    try:
        weighting, rest = (p.strip() for p in name.split(',', 1))
        parts = rest.split()
        rejection = parts[0]
        sigma = float(parts[1]) if len(parts) > 1 else None
        if (weighting not in oe.WEIGHTINGS or rejection not in oe.REJECTIONS
                or (rejection == 'none') != (sigma is None) or len(parts) > 2):
            return None
    except (ValueError, IndexError):
        return None
    spec = make_candidate(weighting, rejection, sigma)
    return spec if spec['name'] == name else None


def candidate_kwargs(spec: Mapping[str, Any], noise_model: NoiseModel | None = None,
                     grad: npt.NDArray[np.float32] | None = None) -> dict[str, Any]:
    """The keywords of `symmetrize_volume` for a candidate dict.

    `noise_model` and `grad` are given only if the candidate needs them
    ('adaptive' weighting or 'model' rejection).
    """
    rejection = spec.get('rejection', 'none')
    kw: dict[str, Any] = {'weighting': spec.get('weighting', 'equal'), 'rejection': rejection,
                          'sigma': None if rejection == 'none' else float(spec.get('sigma', 3.0))}
    for key in ('min_valid', 'poisson_floor'):
        if key in spec:
            kw[key] = spec[key]
    if kw['weighting'] == 'adaptive' or rejection == 'model':
        kw['noise_model'] = noise_model
        if grad is not None:
            kw['grad'] = grad
    return kw


def centred_crop(vol: Any, n: int) -> Any:
    """A box of at most `n` voxels per axis, centred on the voxel nearest hkl = 0.

    The box has the same number of voxels on the two sides of the centre on
    each axis. Thus the symmetry operations map most of the box into the box.
    If hkl = 0 is near the end of an axis, such a box is thin. The box then
    has min(n, axis length) voxels on that axis and contains hkl = 0 as near
    to its centre as possible. If an axis does not contain 0, the box uses
    the middle of that axis.
    """
    from .volume_builder import VolumeData
    sl = []
    for ax in (vol.H, vol.K, vol.L):
        ax = np.asarray(ax, np.float64)
        size = max(1, min(int(n), len(ax)))
        if ax.min() <= 0.0 <= ax.max():
            i0 = int(np.argmin(np.abs(ax)))
            half = max(0, min((int(n) - 1) // 2, i0, len(ax) - 1 - i0))
            if 2 * half + 1 >= size - 1:         # symmetric about hkl = 0
                sl.append(slice(i0 - half, i0 + half + 1))
                continue
            start = min(max(i0 - (size - 1) // 2, 0), len(ax) - size)
        else:
            start = (len(ax) - size) // 2
        sl.append(slice(start, start + size))
    box = tuple(sl)
    return VolumeData(
        intensity=np.ascontiguousarray(vol.intensity[box]),
        H=np.asarray(vol.H)[sl[0]].copy(), K=np.asarray(vol.K)[sl[1]].copy(),
        L=np.asarray(vol.L)[sl[2]].copy(), plane_type=vol.plane_type,
        metadata=dict(vol.metadata),
        counts=None if vol.counts is None else np.ascontiguousarray(vol.counts[box]),
        variance=None if vol.variance is None else np.ascontiguousarray(vol.variance[box]))


def slice_geometry(vol: Any, plane: int, value: float) -> dict[str, Any]:
    """Display geometry of a slice from `extract_volume_slice`.

    The pixel (ix, iy) of the slice has the in-plane Miller indices
    m = m0 + ix * mx + iy * my. For the native plane of the volume, the
    function uses the affine grid of the volume (`volume_affine`). This
    grid includes the shear of an unwarp raster. The other planes are on
    the regular Miller-index axes of the slice.

    With a UB matrix, 'M' maps the in-plane Miller indices to Cartesian
    coordinates in 1/A. 'M' is the inverse of `compute_plane_M_inv`, as in
    the viewer. 'cartesian' is then True. Without a UB matrix, the native
    plane of an unwarp raster uses the inverse of `metadata['M_inv']`.
    Else 'M' is the identity and the axes are in hkl units ('cartesian'
    False).

    Returns {'m0', 'mx', 'my' (2 floats each), 'M' (2 x 2), 'cartesian'}.
    """
    from .rsp_reader import compute_plane_M_inv
    from .volume_builder import volume_affine
    names = ['HK', 'HL', 'KL']
    x_dim, y_dim, f_dim = [(0, 1, 2), (0, 2, 1), (1, 2, 0)][plane]
    axes = [np.asarray(a, np.float64) for a in (vol.H, vol.K, vol.L)]
    if names[plane] == vol.plane_type:
        origin, A = volume_affine(vol)
        i_f = int(np.argmin(np.abs(axes[f_dim] - float(value))))
        base = origin + A[:, f_dim] * i_f
        rows = [x_dim, y_dim]
        m0, mx, my = base[rows], A[rows, x_dim], A[rows, y_dim]
    else:
        xa, ya = axes[x_dim], axes[y_dim]
        dx = (xa[-1] - xa[0]) / (len(xa) - 1) if len(xa) > 1 else 1.0
        dy = (ya[-1] - ya[0]) / (len(ya) - 1) if len(ya) > 1 else 1.0
        m0, mx, my = np.array([xa[0], ya[0]]), np.array([dx, 0.0]), np.array([0.0, dy])
    M, cart = np.eye(2), False
    ub = vol.metadata.get('ub')
    m_inv = vol.metadata.get('M_inv')
    native_raster = (names[plane] == vol.plane_type
                     and vol.metadata.get('grid_kind', 'unwarp_raster') == 'unwarp_raster')
    try:
        if ub is not None:
            M = np.linalg.inv(compute_plane_M_inv(np.asarray(ub, np.float64),
                                                  float(vol.metadata.get('wavelength', 1.0)),
                                                  names[plane]))
            cart = True
        elif m_inv is not None and native_raster:
            M = np.linalg.inv(np.asarray(m_inv, np.float64))
            cart = True
    except (np.linalg.LinAlgError, ValueError):
        cart = False
    if not (cart and np.all(np.isfinite(M))):
        M, cart = np.eye(2), False
    return {'m0': [float(v) for v in m0], 'mx': [float(v) for v in mx],
            'my': [float(v) for v in my], 'M': M.tolist(), 'cartesian': cart}


def count_spread(vol: Any, laue_group: str, n_members: int = 200_000, ratio: float = 10.0,
                 seed: int = 0) -> dict[str, Any]:
    """Spread of the member pixel counts in a sample of orbits, per |q| shell.

    For each sampled voxel (`sample_orbits`), the ratio max N_i / min N_i
    over its finite members with N_i > 0. Returns {'ratio', 'fraction'
    (per shell: the fraction of the sampled voxels with a ratio above
    `ratio`, None for an empty shell), 'total', 'n_voxels'}.
    """
    from .volume_builder import orbit_plan
    plan = orbit_plan(vol, laue_group)
    data = np.ascontiguousarray(vol.intensity, dtype=np.float32)
    counts = np.ascontiguousarray(vol.counts, dtype=np.float32)
    smp = sample_orbits(plan, data, counts, n_members=n_members, seed=seed)
    del counts
    use = np.isfinite(smp.vals) & (smp.ncnt > 0)
    with np.errstate(invalid='ignore', divide='ignore'):
        hi = np.where(use, smp.ncnt, -np.inf).max(axis=0)
        lo = np.where(use, smp.ncnt, np.inf).min(axis=0)
        big = hi / lo > float(ratio)
    frac, nvox = [], []
    for k in range(oe.N_SHELLS):
        m = smp.shell == k
        nvox.append(int(m.sum()))
        frac.append(float(big[m].mean()) if m.any() else None)
    return {'ratio': float(ratio), 'fraction': frac,
            'total': float(big.mean()) if big.size else None, 'n_voxels': nvox}


def _quiet(*_: Any) -> None:
    return None


class Cancelled(Exception):
    """The user cancelled Measure or Compare."""


def _check_cancel(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise Cancelled('Cancelled.')


def measure_volume(vol: Any, laue_group: str, use_gpu: bool | None = None,
                   radius: float = 0.1,
                   log: Callable[[str], None] | None = None,
                   progress: Callable[[int, int], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Step 'Measure' of the averaging panel.

    1. Fits the noise model and keeps G (`fit_noise_model` with
       `return_gradient=True`). If the fit fails, 'model' is None and
       'fit_error' holds the reason.
    2. Symmetrises once with the default options ('equal, mad 3'). This is
       the reference of `bias_checks`. Its diagnostics give chi^2/nu,
       R_sym, the rejected fraction and the Poisson-floor share per shell.
    3. Symmetrises with equal weights and no rejection, and calculates the
       node sum of step 2 divided by the node sum without rejection, per
       shell (`node_sums_by_shell`).
    4. With pixel counts: the spread of the member counts (`count_spread`).

    `cancelled` is a function without arguments. The function calls it
    between the steps and raises `Cancelled` if it returns True.

    Returns a dict. 'reference' is the result of step 2 without variance.
    It is the only full volume that the dict keeps (with G).
    """
    from .volume_builder import HAS_GPU, symmetrize_volume
    log = log or _quiet
    if use_gpu is None:
        use_gpu = HAS_GPU
    times: dict[str, float] = {}
    out: dict[str, Any] = {'laue_group': laue_group, 'shape': list(vol.intensity.shape),
                           'has_counts': vol.counts is not None, 'model': None, 'grad': None,
                           'fit_error': None, 'radius': float(radius)}
    t = time.perf_counter()
    log('Fit the noise model.')
    try:
        out['model'], out['grad'] = fit_noise_model(vol, laue_group, use_gpu=use_gpu,
                                                    return_gradient=True)
    except ValueError as exc:
        out['fit_error'] = str(exc)
    except (np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError,
            RuntimeError) as exc:
        out['fit_error'] = f"{FIT_NUMERIC} ({type(exc).__name__}: {exc})"
    if out['fit_error'] is not None:
        log(f"The noise-model fit failed: {out['fit_error']}")
    times['fit_s'] = time.perf_counter() - t
    _check_cancel(cancelled)
    t = time.perf_counter()
    log("Symmetrise with the default options ('equal, mad 3').")
    ref = symmetrize_volume(vol, laue_group, sigma=3.0, use_gpu=use_gpu,
                            progress_callback=progress)
    times['reference_s'] = time.perf_counter() - t
    out['diagnostics'] = ref.metadata['symmetry_diagnostics']
    out['n_outliers_removed'] = int(ref.metadata.get('n_outliers_removed', 0))
    ref.variance = None                          # keep only the intensity
    out['reference'] = ref
    _check_cancel(cancelled)
    t = time.perf_counter()
    log('Symmetrise with equal weights and no rejection (for the node sums).')
    plain = symmetrize_volume(vol, laue_group, sigma=None, use_gpu=use_gpu,
                              propagate_variance=False, progress_callback=progress)
    out['node_ratio'] = node_sums_by_shell(ref, plain, laue_group, radius)['ratio']
    del plain
    times['no_rejection_s'] = time.perf_counter() - t
    out['count_spread'] = None
    _check_cancel(cancelled)
    if vol.counts is not None:
        t = time.perf_counter()
        out['count_spread'] = count_spread(vol, laue_group)
        times['count_spread_s'] = time.perf_counter() - t
    out['times'] = times
    return out


def compare_candidates(vol: Any, laue_group: str, measured: Mapping[str, Any],
                       candidates: Sequence[Mapping[str, Any]],
                       extra_positions: Iterable[Sequence[float]] = (),
                       slice_plane: int = 0, slice_value: float = 0.0,
                       use_gpu: bool | None = None, n_members: int = 1_000_000,
                       log: Callable[[str], None] | None = None,
                       progress: Callable[[int, int], None] | None = None,
                       cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Step 'Compare' of the averaging panel.

    `measured` is the output of `measure_volume` for the same volume. The
    function scores all candidates (`score_candidates`), symmetrises the
    volume with each candidate, compares it with the reference
    ('equal, mad 3', `bias_checks`) and keeps one slice of it
    (`extract_volume_slice`, plane 0 HK, 1 HL, 2 KL). It keeps no full
    candidate volume. The reference is always in the comparison. The
    suggestion (`suggest`) uses only the candidates with rejection.
    'slice_info' holds the display geometry of the slices (`slice_geometry`).

    Raises ValueError if `measured` has no noise model. Raises `Cancelled`
    if `cancelled()` returns True between two steps or two candidates.
    """
    from .volume_builder import HAS_GPU, extract_volume_slice, symmetrize_volume
    log = log or _quiet
    if use_gpu is None:
        use_gpu = HAS_GPU
    model, grad, ref = measured['model'], measured['grad'], measured['reference']
    if model is None:
        raise ValueError("Compare needs a noise model, and the fit failed. "
                         + str(measured.get('fit_error') or ''))
    cands = [dict(c) for c in candidates]
    if BIAS_REFERENCE not in [c['name'] for c in cands]:
        cands.insert(0, dict(EQUAL_MAD))
    extra = [tuple(float(v) for v in o) for o in extra_positions]
    t = time.perf_counter()
    _check_cancel(cancelled)
    log(f'Score {len(cands)} candidates.')
    scores = score_candidates(vol, laue_group, cands, noise_model=model, grad=grad,
                              n_members=n_members, use_gpu=use_gpu, progress_callback=progress)
    t_score = time.perf_counter() - t
    sl = extract_volume_slice(ref, slice_plane, slice_value)
    slice_info = {'plane': ['HK', 'HL', 'KL'][slice_plane], 'value': sl[6],
                  'x': [float(sl[1][0]), float(sl[1][-1])],
                  'y': [float(sl[2][0]), float(sl[2][-1])],
                  'labels': (sl[3], sl[4], sl[5]),
                  'geometry': slice_geometry(ref, slice_plane, sl[6])}
    slices = {BIAS_REFERENCE: sl[0]}
    bias: dict[str, Any] = {}
    rows = []
    for i, c in enumerate(cands):
        _check_cancel(cancelled)
        name = c['name']
        if name == BIAS_REFERENCE:
            out, dt = ref, float(measured['times']['reference_s'])
        else:
            log(f"Symmetrise with '{name}'.")
            t = time.perf_counter()
            out = symmetrize_volume(vol, laue_group, use_gpu=use_gpu, propagate_variance=False,
                                    progress_callback=progress,
                                    **candidate_kwargs(c, model, grad))
            dt = time.perf_counter() - t
            slices[name] = extract_volume_slice(out, slice_plane, slice_value)[0]
        bias[name] = bias_checks({name: out}, ref, measured.get('radius', 0.1), extra)[name]
        del out
        sc = scores['candidates'][name]
        b = bias[name]
        rows.append({'name': name, 'rejection': c.get('rejection', 'none'),
                     'score': sc['total']['trimmed_mean'], 'node': b['node_sum_ratio'],
                     'extra': dict(b['extra_sum_ratio']), 'diffuse': b['diffuse_sum_ratio'],
                     'lost': b['n_lost'], 'n_no_estimate': sc['n_no_estimate'],
                     'time_s': dt})
    _check_cancel(cancelled)
    eligible ={r['name'] for r in rows if r['rejection'] != 'none'}
    name, reason = suggest({**scores, 'candidates': {k: v for k, v in scores['candidates'].items()
                                                     if k in eligible}},
                           {k: v for k, v in bias.items() if k in eligible})
    skipped = sorted(set(scores['candidates']) - eligible)
    if skipped:
        reason += (" The suggestion never uses the candidates without rejection ("
                   + ', '.join(f"'{s}'" for s in skipped) + ").")
    return {'scores': scores, 'bias': bias, 'rows': rows, 'slices': slices,
            'slice_info': slice_info, 'suggestion': name, 'reason': reason,
            'extra_positions': [list(o) for o in extra],
            'specs': {c['name']: c for c in cands}, 'score_s': t_score}


def _num(x: Any, spec: str = '.4f') -> str:
    return '-' if x is None else format(x, spec)


def format_measure(measured: Mapping[str, Any]) -> str:
    """Text of the Measure results: the noise model and a table per |q| shell."""
    lines = []
    m = measured.get('model')
    if m is None:
        lines.append(f"Noise model: the fit failed. {measured.get('fit_error') or ''}")
    else:
        lines.append(f"Noise model: gain {m.gain:.4g}, rel {m.rel:.4g}, kappa {m.kappa:.4g}, "
                     f"floor {m.floor:.4g} ({m.info.get('n_members', 0):,} members, "
                     f"{m.info.get('iterations', 0)} iterations, "
                     f"{100 * m.info.get('flagged_fraction', 0.0):.2f} % flagged).")
    lines.append(shell_table(measured))
    return '\n'.join(lines)


def shell_rows(measured: Mapping[str, Any]) -> list[list[str]]:
    """Rows of the Measure table: |q| range, chi^2/nu, R_sym, rejected %,
    Poisson floor %, count spread %, node ratio."""
    d = measured['diagnostics']
    sh = d['shells']
    edges = d['shell_edges']
    spread = measured.get('count_spread')
    rows = []
    for i in range(len(edges) - 1):
        rej, flo = sh['rejected_fraction'][i], sh['poisson_floor_fraction'][i]
        sp = spread['fraction'][i] if spread else None
        rows.append([f"{edges[i]:.3f}-{edges[i + 1]:.3f}", _num(sh['chi2_per_nu'][i], '.3f'),
                     _num(sh['r_sym'][i]), _num(None if rej is None else 100 * rej, '.3f'),
                     _num(None if flo is None else 100 * flo, '.1f'),
                     _num(None if sp is None else 100 * sp, '.1f'),
                     _num(measured['node_ratio'][i])])
    return rows


SHELL_HEADERS = ['|q| shell', 'chi2/nu', 'R_sym', 'rejected %', 'floor %', 'spread %',
                 'node ratio']


def shell_table(measured: Mapping[str, Any]) -> str:
    unit = measured['diagnostics'].get('q_unit', '')
    rows = shell_rows(measured)
    head = [f"{SHELL_HEADERS[0]} ({unit})"] + SHELL_HEADERS[1:]
    w = [max(len(head[j]), *(len(r[j]) for r in rows)) for j in range(len(head))]
    out = ['  '.join(h.rjust(w[j]) for j, h in enumerate(head))]
    out += ['  '.join(c.rjust(w[j]) for j, c in enumerate(r)) for r in rows]
    spread = measured.get('count_spread')
    out.append("floor % = voxels where the Poisson floor sets the scale. node ratio = node "
               "sum with MAD 3 / node sum without rejection.")
    if spread:
        out.append(f"spread % = sampled orbits with max/min member pixel count above "
                   f"{spread['ratio']:g}.")
    return '\n'.join(out)


COMPARE_HEADERS = ['candidate', 'score', 'node', 'extra', 'diffuse', 'lost', 'no estimate',
                   'time (s)']


def compare_rows(compared: Mapping[str, Any]) -> list[list[str]]:
    """Rows of the Compare table (strings)."""
    rows = []
    for r in compared['rows']:
        extra = ' '.join(_num(v) for v in r['extra'].values()) or '-'
        rows.append([r['name'], _num(r['score']), _num(r['node']), extra, _num(r['diffuse']),
                     str(r['lost']), str(r['n_no_estimate']), f"{r['time_s']:.1f}"])
    return rows


def format_compare(compared: Mapping[str, Any]) -> str:
    """Text of the Compare results: the table, the suggestion and the limits."""
    rows = compare_rows(compared)
    head = list(COMPARE_HEADERS)
    w = [max(len(head[j]), *(len(r[j]) for r in rows)) for j in range(len(head))]
    out = ['  '.join(h.ljust(w[j]) if j == 0 else h.rjust(w[j]) for j, h in enumerate(head))]
    out += ['  '.join(c.ljust(w[j]) if j == 0 else c.rjust(w[j]) for j, c in enumerate(r))
            for r in rows]
    if compared['extra_positions']:
        out.append('extra = sum ratio at ' + ', '.join(_offset_label(o) for o in
                                                         compared['extra_positions']) + '.')
    out.append("Ratios are sums divided by the sums of 'equal, mad 3'. Lower scores are better.")
    out.append(f"Suggestion: {compared['suggestion']}. {compared['reason']}")
    out.append(LIMITS_TEXT)
    return '\n'.join(out)


def choice_record(name: str, reason: str, measured: Mapping[str, Any] | None,
                  compared: Mapping[str, Any] | None, region: str,
                  split: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """The JSON-able record of an averaging choice for `metadata['averaging_choice']`.

    `split` is the output of `compare_split_halves`, or None. With it, each
    candidate of the record also gets its split-half difference.
    """
    spec = ((compared or {}).get('specs', {}).get(name) or known_candidate(name) or {})
    rec: dict[str, Any] = {
        'name': name, 'weighting': spec.get('weighting'), 'rejection': spec.get('rejection'),
        'sigma': spec.get('sigma'), 'reason': reason,
        'suggested': (compared or {}).get('suggestion'), 'compared_on': region,
        'fit_error': (measured or {}).get('fit_error'),
        'noise_model': None, 'extra_positions': (compared or {}).get('extra_positions', []),
        'candidates': {}}
    model = (measured or {}).get('model')
    if model is not None:
        rec['noise_model'] = {'gain': model.gain, 'rel': model.rel, 'kappa': model.kappa,
                              'floor': model.floor}
    for r in (compared or {}).get('rows', []):
        rec['candidates'][r['name']] = {
            'score': r['score'], 'node_sum_ratio': r['node'], 'extra_sum_ratio': r['extra'],
            'diffuse_sum_ratio': r['diffuse'], 'lost_voxels': r['lost'],
            'n_no_estimate': r['n_no_estimate'], 'time_s': r['time_s']}
    if split:
        for cname, res in split['rows'].items():
            rec['candidates'].setdefault(cname, {})['split_half'] = res
        rec['split_half'] = {'definition': SPLIT_HALF_DEFINITION,
                             'halves_rule': split.get('halves_rule'),
                             'shell_edges': split['shell_edges'], 'q_unit': split['q_unit']}
    return rec


# ──────────────────────────────────────────────────────────────────
# Split halves
# ──────────────────────────────────────────────────────────────────

SPLIT_HALF_DEFINITION = ('D = sum|A - B| / sum(|A| + |B|) over the voxels that are finite in A '
                         'and B. A and B are the odd-frame and the even-frame volumes. Each half '
                         'gets its own noise model and is symmetrised with the candidate. '
                         "'node' = voxels within 0.1 (hkl distance) of an integer node, "
                         "'diffuse' = the other voxels.")

# Frame map of the frame tools (rawrecon.frames). The text is here so that the
# Averaging panel can use it without an import of rawrecon.
FRAME_MAP_FLOOR = 1e-3      # value of a cell without observed counts (below 1 / 1.5)
FRAME_MAP_DEFINITION = (
    "Each cell of the frame map is the observed / predicted ratio of one frame and 5-degree "
    "sector, divided by the median of this ratio over all frames of the same sector. "
    "Thus 1 means that the frame agrees with the other frames in this sector. The sector "
    "median removes a pattern that is the same in all frames (for example a shadow). "
    f"A cell without observed counts gets the value {FRAME_MAP_FLOOR:g} (an empty frame). "
    "A frame is off if the median over its sectors differs from 1 by more than 10 %.")


def split_half_masks(vol: Any, laue_group: str,
                     radius: float = 0.1) -> tuple[npt.NDArray[np.int8], npt.NDArray[np.bool_],
                                                   list[float], str]:
    """|q| shell index (int8, 0 to N_SHELLS - 1) and node mask of each voxel of `vol`.

    Also returns the shell edges and the |q| unit. A node voxel is at most
    `radius` (hkl distance) from the nearest integer node.
    """
    from .volume_builder import orbit_plan, volume_affine
    plan = orbit_plan(vol, laue_group)
    origin, A = volume_affine(vol)
    shape = vol.intensity.shape
    shell = np.empty(shape, np.int8)
    node = np.empty(shape, bool)
    c1 = np.broadcast_to(np.arange(shape[1])[:, None], shape[1:])
    c2 = np.broadcast_to(np.arange(shape[2])[None, :], shape[1:])
    for h0 in range(shape[0]):
        c0 = np.full(shape[1:], h0)
        shell[h0] = oe._shell_of(plan, c0, c1, c2)
        d2 = np.zeros(shape[1:])
        for d in range(3):
            f = origin[d] + A[d, 0] * h0 + A[d, 1] * c1 + A[d, 2] * c2
            d2 += (f - np.rint(f)) ** 2
        node[h0] = d2 <= radius * radius
    edges = [plan.q_max * i / oe.N_SHELLS for i in range(oe.N_SHELLS + 1)]
    return shell, node, edges, plan.q_unit


def split_half_difference(a: npt.NDArray[np.floating], b: npt.NDArray[np.floating],
                          shell: npt.NDArray[np.int8],
                          node: npt.NDArray[np.bool_]) -> dict[str, Any]:
    """Split-half difference of two volumes (see `SPLIT_HALF_DEFINITION`).

    Returns 'total', 'node', 'diffuse', 'shells' (one value per |q| shell,
    None for a shell without common voxels) and 'n_both'.
    """
    both = np.isfinite(a) & np.isfinite(b)
    num = np.abs(a.astype(np.float64) - b)
    den = np.abs(a.astype(np.float64)) + np.abs(b)

    def ratio(m: npt.NDArray[np.bool_]) -> float | None:
        d = float(den[m].sum())
        return float(num[m].sum()) / d if d > 0 else None
    return {'total': ratio(both), 'node': ratio(both & node), 'diffuse': ratio(both & ~node),
            'shells': [ratio(both & (shell == k)) for k in range(oe.N_SHELLS)],
            'n_both': int(both.sum())}


def compare_split_halves(odd: Any, even: Any, laue_group: str,
                         candidates: Sequence[Mapping[str, Any]],
                         use_gpu: bool | None = None,
                         log: Callable[[str], None] | None = None,
                         progress: Callable[[int, int], None] | None = None,
                         cancelled: Callable[[], bool] | None = None) -> dict[str, Any]:
    """Split-half difference of each candidate (see `SPLIT_HALF_DEFINITION`).

    `odd` and `even` are two volumes on the same grid (the two halves of one
    dataset). Each half gets its own noise model (`_fit`), because the gain
    of a half is not the gain of the full volume. A candidate that needs a
    noise model gets None if the fit of a half fails.
    Returns 'rows' ({name: `split_half_difference` or None}), 'shell_edges',
    'q_unit', 'fit_errors' and 'time_s'. Raises `Cancelled` between two
    candidates if `cancelled()` returns True.
    """
    from .volume_builder import HAS_GPU, symmetrize_volume
    log = log or _quiet
    if use_gpu is None:
        use_gpu = HAS_GPU
    if odd.intensity.shape != even.intensity.shape:
        raise ValueError('The two halves must have the same grid.')
    t0 = time.perf_counter()
    shell, node, edges, unit = split_half_masks(odd, laue_group)
    fits: dict[str, Any] = {}
    errors: dict[str, str | None] = {}
    needs_model = any(c.get('weighting') == 'adaptive' or c.get('rejection') == 'model'
                      for c in candidates)
    for name, half in (('odd', odd), ('even', even)):
        _check_cancel(cancelled)
        fits[name], errors[name] = (None, None), None
        if not needs_model:
            continue
        log(f'Fit the noise model of the {name} half.')
        try:
            fits[name] = _fit(half, laue_group, None, None, use_gpu=use_gpu)
        except (ValueError, np.linalg.LinAlgError, FloatingPointError, ZeroDivisionError,
                RuntimeError) as exc:
            errors[name] = str(exc)
            log(f'The noise-model fit of the {name} half failed: {exc}')
    rows: dict[str, Any] = {}
    for i, c in enumerate(candidates):
        _check_cancel(cancelled)
        name = c['name']
        need = c.get('weighting') == 'adaptive' or c.get('rejection') == 'model'
        if need and (fits['odd'][0] is None or fits['even'][0] is None):
            rows[name] = None
            continue
        log(f"Split halves: symmetrise the two halves with '{name}'.")
        out = []
        for half, (model, grad) in ((odd, fits['odd']), (even, fits['even'])):
            out.append(symmetrize_volume(half, laue_group, use_gpu=use_gpu,
                                         propagate_variance=False,
                                         **candidate_kwargs(c, model, grad)).intensity)
        rows[name] = split_half_difference(out[0], out[1], shell, node)
        del out
        if progress:
            progress(i + 1, len(candidates))
    return {'rows': rows, 'shell_edges': edges, 'q_unit': unit, 'fit_errors': errors,
            'time_s': time.perf_counter() - t0}
