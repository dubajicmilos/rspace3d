"""
orbit_engine.py — Per-voxel orbit engine of `symmetrize_volume`.

Each output voxel is processed on its own. The engine gathers the orbit
members of the voxel, finds the median and the MAD of the finite members,
rejects outliers, and writes the mean of the members that remain. It also
propagates the variance of each voxel and collects diagnostics per |q| shell.

There are three implementations of the same algorithm:

* `run_gpu`: a CUDA kernel (`cupy.RawKernel`) with one thread per voxel.
* `run_cpu` with numba (optional, `_orbit_numba`): the same per-voxel code
  on all CPU cores. `run_cpu` uses it when numba is installed.
* `run_cpu` with NumPy: vectorised over boxes of voxels.

All use the same operations in the same order. Thus their intensity,
variance and outlier count are bit-identical. The CUDA code is compiled with
`--fmad=false`, so that no multiply-add is fused, and it uses correctly
rounded division and square root. Without a variance, the NumPy engine
calculates the interpolated members with `map_coordinates`, as rspace3d 2.2
did. This gives the same bits as the CUDA code.

Kinds of orbit member (see `volume_builder.index_space_ops`):

* 'perm' and 'nearest': a 1D table per source axis gives the source index
  (`np.rint`, -1 outside the grid).
* 'integer': int64 index arithmetic.
* 'interp': trilinear interpolation of the measured neighbours. The float64
  arithmetic is the same as in `scipy.ndimage.map_coordinates(order=1,
  mode='constant')`.

The diagnostics are not part of the bit-identity. The GPU adds the float
sums in a different order.

Each voxel of an orbit sees the same members. Thus the float sums of the
diagnostics (chi^2, nu and the R_sym sums) of a voxel are divided by the
number of distinct sources among its finite members. An exact orbit then
counts once. With interpolated members the members of two orbit voxels are
not exactly the same, and this weight is approximate.
"""

from __future__ import annotations

import itertools
import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np
import numpy.typing as npt
from scipy.ndimage import map_coordinates

MAX_OPS = 48
N_SHELLS = 10
MAD_SCALE = 1.4826
_KIND_CODE = {'perm': 0, 'nearest': 0, 'integer': 1, 'interp': 2}
_DEN_MIN = np.float32(1e-6)   # support weight below which an interp member is missing

# Integer diagnostic sums per shell (columns of `OrbitResult.int_sums`)
_INT_KEYS = ('n_voxels', 'n_members', 'n_rejected', 'n_tested', 'n_poisson_floor',
             'n_outliers_removed')
# Float diagnostic sums per shell (columns of `OrbitResult.float_sums`). Each
# voxel adds its value divided by the number of distinct sources of its orbit.
_FLOAT_KEYS = ('chi2', 'nu', 'rsym_num', 'rsym_den')
_NI = len(_INT_KEYS)
_NF = len(_FLOAT_KEYS)
_I_OUT = _INT_KEYS.index('n_outliers_removed')


@dataclass
class OrbitPlan:
    """Operation tables and shell geometry for one volume shape."""
    shape: tuple[int, int, int]
    n_ops: int
    identity_index: int
    kind: npt.NDArray[np.int32]        # (n_ops,) 0 table, 1 integer, 2 interp
    opi: npt.NDArray[np.int32]         # (n_ops, 8) kind, target axis j[d], table offset[d]
    opl: npt.NDArray[np.int64]         # (n_ops, 12) integer M (row major) and t
    opd: npt.NDArray[np.float64]       # (n_ops, 12) float64 M (row major) and t
    tab: npt.NDArray[np.int64]         # concatenated 1D source-index tables
    geo: npt.NDArray[np.float64]       # (22,) origin, A, G (row major), shells / q_max
    q_max: float
    q_unit: str
    edge_tol: float
    max_index_error: float


@dataclass
class OrbitResult:
    """Output of `run_cpu` and `run_gpu`."""
    intensity: npt.NDArray[np.float32]
    variance: npt.NDArray[np.float32] | None
    n_outliers_removed: int
    int_sums: npt.NDArray[np.int64]      # (N_SHELLS, len(_INT_KEYS))
    float_sums: npt.NDArray[np.float64]  # (N_SHELLS, len(_FLOAT_KEYS))
    engine: str
    timing: dict[str, float] = field(default_factory=dict)


# ──────────────────────────────────────────────────────────────────
# Plan
# ──────────────────────────────────────────────────────────────────

def build_plan(shape: tuple[int, int, int], ops: list[dict[str, Any]], identity_index: int,
               origin: npt.NDArray[np.float64], A: npt.NDArray[np.float64],
               G: npt.NDArray[np.float64], q_unit: str, edge_tol: float) -> OrbitPlan:
    """Make the operation tables for `run_cpu` and `run_gpu`.

    `ops` is the list from `index_space_ops`. `origin` and `A` give the Miller
    index of a voxel (`hkl = origin + A @ i`). `G` is the reciprocal metric
    tensor, so that |q|^2 = hkl^T G hkl.
    """
    shape = tuple(int(s) for s in shape)
    n_ops = len(ops)
    if n_ops > MAX_OPS:
        raise ValueError(f"at most {MAX_OPS} operations are supported, got {n_ops}")
    kind = np.zeros(n_ops, np.int32)
    opi = np.zeros((n_ops, 8), np.int32)
    opl = np.zeros((n_ops, 12), np.int64)
    opd = np.zeros((n_ops, 12), np.float64)
    tables = []
    offset = 0
    max_err = 0.0
    for o, op in enumerate(ops):
        code = _KIND_CODE[op['kind']]
        kind[o] = code
        opi[o, 0] = code
        M = np.asarray(op['M'])
        t = np.asarray(op['t'])
        if code == 0:
            # Same as rspace3d <= 2.2 `_gather_perm`: rint of the float64
            # position, -1 where the source is outside the grid.
            for d in range(3):
                j = int(np.nonzero(M[d])[0][0])
                m = np.arange(shape[j], dtype=np.float64)
                pos = float(M[d, j]) * m + float(t[d])
                src = np.rint(pos).astype(np.int64)
                ok = (src >= 0) & (src < shape[d])
                if ok.any():
                    max_err = max(max_err, float(np.abs(pos - src)[ok].max()))
                tables.append(np.where(ok, src, -1))
                opi[o, 1 + d] = j
                opi[o, 4 + d] = offset
                offset += shape[j]
        elif code == 1:
            opl[o, :9] = np.asarray(M, dtype=np.int64).ravel()
            opl[o, 9:] = np.asarray(t, dtype=np.int64)
        else:
            opd[o, :9] = np.asarray(M, dtype=np.float64).ravel()
            opd[o, 9:] = np.asarray(t, dtype=np.float64)
    if offset >= 2**31:
        raise ValueError("operation tables are too large")
    tab = np.concatenate(tables) if tables else np.zeros(1, np.int64)

    origin = np.asarray(origin, np.float64)
    A = np.asarray(A, np.float64)
    G = np.asarray(G, np.float64)
    corners = np.array([[i, j, k] for i in (0, shape[0] - 1) for j in (0, shape[1] - 1)
                        for k in (0, shape[2] - 1)], dtype=np.float64)
    # |q| is convex in the array index, so its maximum is at a corner
    q_max = float(max(_q_of(origin, A, G, c[0], c[1], c[2]) for c in corners))
    geo = np.zeros(22, np.float64)
    geo[0:3] = origin
    geo[3:12] = A.ravel()
    geo[12:21] = G.ravel()
    geo[21] = N_SHELLS / q_max if q_max > 0 else 0.0
    return OrbitPlan(shape=shape, n_ops=n_ops, identity_index=int(identity_index),
                     kind=kind, opi=opi, opl=opl, opd=opd, tab=tab, geo=geo,
                     q_max=q_max, q_unit=q_unit, edge_tol=float(edge_tol),
                     max_index_error=max_err)


def _q_of(origin: Any, A: Any, G: Any, c0: Any, c1: Any, c2: Any) -> Any:
    """|q| of the voxel (c0, c1, c2). The CUDA kernel uses the same operation order."""
    h = [origin[d] + ((A[d, 0] * c0 + A[d, 1] * c1) + A[d, 2] * c2) for d in range(3)]
    g = [(G[d, 0] * h[0] + G[d, 1] * h[1]) + G[d, 2] * h[2] for d in range(3)]
    q2 = (h[0] * g[0] + h[1] * g[1]) + h[2] * g[2]
    return np.sqrt(np.maximum(q2, 0.0))


def _shell_of(plan: OrbitPlan, c0: Any, c1: Any, c2: Any) -> npt.NDArray[np.int64]:
    geo = plan.geo
    q = _q_of(geo[0:3], geo[3:12].reshape(3, 3), geo[12:21].reshape(3, 3),
              c0.astype(np.float64), c1.astype(np.float64), c2.astype(np.float64))
    return np.minimum((q * geo[21]).astype(np.int64), N_SHELLS - 1)


# ──────────────────────────────────────────────────────────────────
# CPU engine (NumPy)
# ──────────────────────────────────────────────────────────────────

_BOX_VALUES = 800_000          # n_ops x voxels of one NumPy box
_SLAB_VOXELS = 1 << 22         # voxels of one slab for the collision test


def _boxes(shape: tuple[int, int, int], target: int) -> Any:
    """Boxes (h0, h1, k0, k1) of about `target` voxels, in C order.

    Each box has all l. It has all k, or it has one h-layer. Thus the voxels
    of a box are one contiguous range of the flat volume.
    """
    nh, nk, nl = shape
    if nk * nl <= target:
        rows = max(1, target // max(nk * nl, 1))
        for h0 in range(0, nh, rows):
            yield h0, min(nh, h0 + rows), 0, nk
    else:
        cols = max(1, target // max(nl, 1))
        for h0 in range(nh):
            for k0 in range(0, nk, cols):
                yield h0, h0 + 1, k0, min(nk, k0 + cols)


def _boxes_in(slab: tuple[int, int, int, int], nk: int, nl: int, target: int) -> Any:
    """Boxes of about `target` voxels in one slab of `_boxes`, in C order."""
    h0, h1, k0, k1 = slab
    if k0 == 0 and k1 == nk:
        for a, b, kk0, kk1 in _boxes((h1 - h0, nk, nl), target):
            yield h0 + a, h0 + b, kk0, kk1
    else:                                    # one h-layer and part of the k range
        cols = max(1, target // max(nl, 1))
        for kk in range(k0, k1, cols):
            yield h0, h1, kk, min(k1, kk + cols)


def _along(j: int, a: npt.NDArray[Any]) -> npt.NDArray[Any]:
    """1D array `a` with the shape that broadcasts along axis j of a box."""
    shape = [1, 1, 1]
    shape[j] = a.shape[0]
    return a.reshape(shape)


@dataclass
class _Source:
    """Input arrays of the NumPy engine."""
    data: npt.NDArray[np.float32]            # (nh, nk, nl)
    flat: npt.NDArray[np.float32]            # data.reshape(-1)
    var: npt.NDArray[np.float32] | None      # stored variance (var_mode 1)
    var_flat: npt.NDArray[np.float32] | None
    var_mode: int                            # 0 none, 1 stored, 2 derived max(x, 1)
    filled: npt.NDArray[np.float32] | None   # data with NaN = 0 (interp, var_mode 0)
    weight: npt.NDArray[np.float32] | None   # 1 where data is finite, else 0


def _table_axes(plan: OrbitPlan, o: int, r: list[Any]) -> list[Any]:
    """Source index on each axis of a 'perm' or 'nearest' member (-1 outside the grid).

    Each index is a 3D array that broadcasts to the box. It comes from a 1D table.
    """
    sd = []
    for d in range(3):
        j = int(plan.opi[o, 1 + d])
        sd.append(_along(j, plan.tab[int(plan.opi[o, 4 + d]) + r[j]]))
    return sd


def _integer_axes(plan: OrbitPlan, o: int, c: list[Any]) -> list[Any]:
    """Source index on each axis of an 'integer' member (int64, can be outside the grid).

    Each index is calculated only on the axes that it depends on.
    """
    m = plan.opl[o]
    sd = []
    for d in range(3):
        s = None
        for j in range(3):
            if m[3 * d + j] != 0:
                term = m[3 * d + j] * c[j]
                s = term if s is None else s + term
        sd.append(np.full((1, 1, 1), m[9 + d], np.int64) if s is None else s + m[9 + d])
    return sd


def _interp_axes(plan: OrbitPlan, o: int, dc: list[Any]) -> tuple[Any, ...]:
    """Source coordinates and trilinear weights of an 'interp' member.

    The source coordinate on axis d is ((m0 c0 + m1 c1) + m2 c2) + t, as in
    the CUDA kernel. A term with m = 0 is left out. It adds a zero, and the
    sign of a zero coordinate does not change the result. Thus each
    coordinate is calculated only on the axes that it depends on. A
    coordinate less than `edge_tol` outside the grid moves to the grid edge.

    Returns the lists (one entry per axis) of the coordinates, the weights of
    the lower and of the upper neighbour, the lower neighbour index and the
    flags 'inside the grid', and the flag 'inside on all axes'.
    """
    sizes = plan.shape
    m = plan.opd[o]
    cs, w0, w1, base, ins_d = [], [], [], [], []
    inside = None
    for d in range(3):
        cd = None
        for j in range(3):
            if m[3 * d + j] != 0.0:
                term = m[3 * d + j] * dc[j]
                cd = term if cd is None else cd + term
        cd = np.full((1, 1, 1), m[9 + d]) if cd is None else cd + m[9 + d]
        hi = float(sizes[d] - 1)
        cd = np.where((cd < 0) & (cd > -plan.edge_tol), 0.0, cd)
        cd = np.where((cd > hi) & (cd < hi + plan.edge_tol), hi, cd)
        ins = (cd >= 0) & (cd <= hi)
        inside = ins if inside is None else inside & ins
        f = np.floor(cd)
        a0 = 1.0 - (cd - f)
        cs.append(cd)
        w0.append(a0)
        w1.append(1.0 - a0)
        base.append(np.where(ins, f, 0).astype(np.int64))
        ins_d.append(ins)
    return cs, w0, w1, base, ins_d, inside


def _sources(plan: OrbitPlan, r: list[Any], c: list[Any], dc: list[Any]) -> list[list[Any]]:
    """Source index on each axis of each member, for `_collision_voxels`.

    An 'interp' member has the grid index on axis d where its coordinate is
    on the grid, else -1. It reads one source voxel only where all three
    coordinates are on the grid.
    """
    out = []
    for o in range(plan.n_ops):
        kind = plan.kind[o]
        if kind == 0:
            out.append(_table_axes(plan, o, r))
        elif kind == 1:
            out.append(_integer_axes(plan, o, c))
        else:
            _, _, w1, base, ins_d, _ = _interp_axes(plan, o, dc)
            out.append([np.where((w1[d] == 0) & ins_d[d], base[d], -1) for d in range(3)])
    return out


def _member_table(plan: OrbitPlan, o: int, src: _Source, r: list[Any]) -> tuple[Any, Any]:
    """'perm' and 'nearest' member: values and variances (None without variance)."""
    sd = _table_axes(plan, o, r)
    ok = (sd[0] >= 0) & (sd[1] >= 0) & (sd[2] >= 0)
    ix = [np.maximum(s, 0) for s in sd]
    x = src.data[ix[0], ix[1], ix[2]]
    bad = ~np.isfinite(x)
    bad |= ~ok
    x[bad] = np.nan
    xv = None
    if src.var_mode == 1:
        xv = src.var[ix[0], ix[1], ix[2]]
    elif src.var_mode == 2:
        xv = np.maximum(x, np.float32(1.0))
    return x, xv


def _member_integer(plan: OrbitPlan, o: int, src: _Source, c: list[Any],
                    shape: tuple[int, int, int]) -> tuple[Any, Any]:
    """'integer' member (int64 index arithmetic): values and variances."""
    nh, nk, nl = plan.shape
    sd = _integer_axes(plan, o, c)
    ok = None
    for d in range(3):
        okd = (sd[d] >= 0) & (sd[d] < plan.shape[d])
        ok = okd if ok is None else ok & okd
    ok = np.broadcast_to(ok, shape)
    key = np.where(ok, (sd[0] * nk + sd[1]) * nl + sd[2], 0)
    x = src.flat[key]
    x[~(ok & np.isfinite(x))] = np.nan
    xv = None
    if src.var_mode == 1:
        xv = src.var_flat[key]
    elif src.var_mode == 2:
        xv = np.maximum(x, np.float32(1.0))
    return x, xv


def _member_interp(plan: OrbitPlan, o: int, src: _Source, dc: list[Any],
                   shape: tuple[int, int, int]) -> tuple[Any, Any]:
    """'interp' member: trilinear interpolation of the measured neighbours.

    Returns the values and the variances (None without variance). The
    variance is sum_k (w_k / W)^2 var_k over the measured corners k.
    """
    nh, nk, nl = plan.shape
    sizes = plan.shape
    cs, w0, w1, base, ins_d, inside = _interp_axes(plan, o, dc)
    nan = np.float32(np.nan)

    if src.var_mode == 0 and src.filled is not None:
        # The same float64 arithmetic in compiled code (see the module docstring)
        coords = np.stack([np.broadcast_to(cd, shape).ravel() for cd in cs])
        num32 = map_coordinates(src.filled, coords, order=1, mode='constant', cval=0.0)
        den32 = map_coordinates(src.weight, coords, order=1, mode='constant', cval=0.0)
        del coords
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            x = np.where(den32 > _DEN_MIN, num32 / np.maximum(den32, _DEN_MIN), nan)
        x[~np.isfinite(x)] = nan
        return x.reshape(shape), None

    num = np.zeros(shape, np.float64)
    den = np.zeros(shape, np.float64)
    taps = []
    with np.errstate(invalid='ignore', over='ignore'):
        for s in itertools.product((0, 1), repeat=3):
            a = [w1[d] if s[d] else w0[d] for d in range(3)]
            nonzero = [a[d] != 0 for d in range(3)]
            if not all(z.any() for z in nonzero):
                continue                     # this corner has the weight 0 everywhere
            idx = [np.minimum(base[d] + s[d], sizes[d] - 1) for d in range(3)]
            fl = (idx[0] * nk + idx[1]) * nl + idx[2]
            xx = src.flat[fl]
            use = inside & np.isfinite(xx)
            for z in nonzero:
                if not z.all():
                    use = use & z
            use = np.broadcast_to(use, shape)
            # ((x a0) a1) a2 as in the CUDA kernel. A factor 1.0 is exact and is left out.
            p = xx.astype(np.float64)
            for d in range(3):
                if np.all(a[d] == 1.0):
                    continue
                if np.broadcast_shapes(p.shape, a[d].shape) == p.shape:
                    np.multiply(p, a[d], out=p)
                else:
                    p = p * a[d]
            np.add(num, p, out=num, where=use)
            del p
            w = (a[0] * a[1]) * a[2]
            np.add(den, w, out=den, where=use)
            if src.var_mode != 0:
                taps.append((fl, w, use, xx))
    num32 = num.astype(np.float32)
    den32 = den.astype(np.float32)
    del num
    good = inside & (den32 > _DEN_MIN)
    with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
        x = np.where(good, num32 / np.maximum(den32, _DEN_MIN), nan)
    x[~np.isfinite(x)] = nan
    xv = None
    if src.var_mode != 0:
        vs = np.zeros(shape, np.float64)
        rr = np.empty(shape, np.float64)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            for fl, w, use, xx in taps:
                vk = src.var_flat[fl] if src.var_mode == 1 else np.maximum(xx, np.float32(1.0))
                np.divide(w, den, out=rr)                 # r = w / W
                np.multiply(rr, rr, out=rr)
                np.multiply(rr, vk, out=rr)               # (r r) var, var exact in float64
                np.add(vs, rr, out=vs, where=use)
        xv = vs.astype(np.float32)
    return x, xv


def _median_sorted(s: npt.NDArray[np.float32], n: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Median of the first n entries of each row of `s` (rows sorted, NaN last)."""
    last = s.shape[1] - 1
    h = n // 2
    hi = np.take_along_axis(s, np.minimum(h, last)[:, None], 1)[:, 0]
    lo = np.take_along_axis(s, np.clip(h - 1, 0, last)[:, None], 1)[:, 0]
    return np.where(n % 2 == 1, hi, (lo + hi) * np.float32(0.5))


def _row_median(a: npt.NDArray[np.float32], n: npt.NDArray[np.int64]) -> npt.NDArray[np.float32]:
    """Median over axis 0 of `a` (n_ops, count) of the n finite entries of each column."""
    s = np.ascontiguousarray(a.T)
    s.sort(axis=1)                         # NaN last
    return _median_sorted(s, n)


def _multiplicity(keys: npt.NDArray[np.int64],
                  surv: npt.NDArray[np.bool_]) -> tuple[Any, Any]:
    """Multiplicity of each survivor's source and first-occurrence flags.

    `keys` and `surv` are (n_ops, count). A source that occurs c times among
    the survivors of a voxel gives c to each of these survivors. The first
    survivor of each source in operation order is flagged.
    """
    n_ops = keys.shape[0]
    sentinel = -1000 - np.arange(n_ops, dtype=np.int64)[:, None]
    k = np.ascontiguousarray(np.where(surv, keys, sentinel).T)
    order = np.argsort(k, axis=1, kind='stable')
    ks = np.take_along_axis(k, order, 1)
    start = np.ones(ks.shape, bool)
    start[:, 1:] = ks[:, 1:] != ks[:, :-1]
    run = np.cumsum(start.ravel()) - 1
    lengths = np.bincount(run)
    mult_sorted = lengths[run].reshape(ks.shape)
    mult = np.empty_like(mult_sorted)
    np.put_along_axis(mult, order, mult_sorted, 1)
    first = np.empty_like(start)
    np.put_along_axis(first, order, start, 1)
    return mult.T, first.T & surv


def _collision_voxels(sources: list[list[Any]],
                      shape: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """Flat indices (in the box) of the voxels where two members can read the same source.

    `sources[o]` holds the source index of member o on each axis (3D arrays
    that broadcast to the box, negative where the member does not read one
    source voxel). Two members read the same source only where their
    indices are equal on all three axes. The test uses the small broadcast
    arrays, thus it is fast. The result can contain more voxels than
    necessary, but it contains all voxels where two finite members read the
    same source.
    """
    mask = None
    n_ops = len(sources)
    for o in range(n_ops):
        so = sources[o]
        for p in range(o + 1, n_ops):
            sp = sources[p]
            parts: list[Any] | None = []
            for d in range(3):
                e = (so[d] == sp[d]) & (so[d] >= 0)
                if not e.any():
                    parts = None
                    break
                if not e.all():
                    parts.append(e)
            if parts is None:
                continue
            if mask is None:
                mask = np.zeros(shape, bool)
            if not parts:
                mask[...] = True
                continue
            hit = parts[0]
            for e in parts[1:]:
                hit = hit & e
            idx = np.nonzero(hit)
            if idx[0].size == 0:
                continue
            # an axis of length 1 in `hit` means all indices on that axis
            mask[tuple(slice(None) if hit.shape[d] == 1 else idx[d] for d in range(3))] = True
    if mask is None:
        return np.zeros(0, np.int64)
    return np.flatnonzero(mask)


def _keys_at(sources: list[list[Any]], shape: tuple[int, int, int],
             cand: npt.NDArray[np.int64], grid: tuple[int, int, int]) -> npt.NDArray[np.int64]:
    """(n_ops, len(cand)) flat source index of each member at the voxels `cand`.

    A member that does not read one source voxel gets the unique key -1 - o.
    """
    nh, nk, nl = grid
    at = np.unravel_index(cand, shape)
    keys = np.empty((len(sources), cand.size), np.int64)
    for o, sd in enumerate(sources):
        s = [np.broadcast_to(sd[d], shape)[at] for d in range(3)]
        one = (s[0] >= 0) & (s[1] >= 0) & (s[2] >= 0)
        keys[o] = np.where(one, (s[0] * nk + s[1]) * nl + s[2], -1 - o)
    return keys


def _ranges(h0: int, h1: int, k0: int, k1: int, nl: int) -> tuple[list[Any], ...]:
    """Index ranges of a box: 1D int64 (r), 3D int64 (c) and 3D float64 (dc)."""
    r = [np.arange(h0, h1, dtype=np.int64), np.arange(k0, k1, dtype=np.int64),
         np.arange(nl, dtype=np.int64)]
    c = [_along(j, r[j]) for j in range(3)]
    dc = [ci.astype(np.float64) for ci in c]
    return r, c, dc


def _box_cpu(plan: OrbitPlan, src: _Source, h0: int, h1: int, k0: int, k1: int,
             sigma: float | None, min_valid: int, poisson_floor: bool,
             cand: npt.NDArray[np.int64], keys_c: npt.NDArray[np.int64] | None) -> tuple[Any, ...]:
    """Output, output variance and diagnostic sums of the voxels of one box.

    `cand` holds the flat indices (in the box) of the voxels where two
    members can read the same source (`_collision_voxels`), and `keys_c`
    the source keys of all members at these voxels (`_keys_at`).
    """
    nh, nk, nl = plan.shape
    n_ops = plan.n_ops
    shape = (h1 - h0, k1 - k0, nl)
    count = shape[0] * shape[1] * shape[2]
    r, c, dc = _ranges(h0, h1, k0, k1, nl)
    var_mode = src.var_mode
    has_var = var_mode != 0

    vals = np.empty((n_ops, count), np.float32)
    mvar = np.empty((n_ops, count), np.float32) if has_var else None
    for o in range(n_ops):
        kind = plan.kind[o]
        if kind == 0:
            x, xv = _member_table(plan, o, src, r)
        elif kind == 1:
            x, xv = _member_integer(plan, o, src, c, shape)
        else:
            x, xv = _member_interp(plan, o, src, dc, shape)
        vals[o] = np.broadcast_to(x, shape).reshape(-1)
        if has_var:
            mvar[o] = np.broadcast_to(xv, shape).reshape(-1)
        del x, xv
    fin = np.isfinite(vals)
    n = fin.sum(axis=0)

    tested = np.zeros(count, bool)
    floor_used = np.zeros(count, bool)
    surv = fin
    rej = None
    if sigma is not None:
        with np.errstate(invalid='ignore'):
            med = _row_median(vals, n)
            dev = np.abs(vals - med)
            scale = _row_median(dev, n) * np.float32(MAD_SCALE)
            if poisson_floor:
                root = np.sqrt(np.maximum(med, np.float32(1.0)))
                floor_used = root > scale
                scale = np.maximum(scale, root)
            thr = np.float32(sigma) * scale
            tested = (n >= min_valid) & (n > 0)
            rej = (dev > thr) & tested
        del dev, med, scale, thr
        floor_used &= tested
        surv = fin & ~rej
    m = surv.sum(axis=0)

    acc = np.zeros(count, np.float32)
    for i in range(n_ops):
        np.add(acc, vals[i], out=acc, where=surv[i])
    with np.errstate(invalid='ignore', divide='ignore'):
        out = np.where(m > 0, acc / m.astype(np.float32), np.float32(np.nan))
    mean = out.astype(np.float64)
    tmp = np.empty(count, np.float64)

    # Two members can read the same source only at the voxels in `cand`.
    # Elsewhere each source has the multiplicity 1, and the number of
    # distinct sources is the number of finite members.
    n_src = n.astype(np.float64)
    if cand.size:
        _, first_fin = _multiplicity(keys_c, fin[:, cand])
        n_src[cand] = first_fin.sum(axis=0)
        del first_fin

    out_var = None
    chi = nd = None
    if has_var:
        m64 = m.astype(np.float64)
        accv = np.zeros(count, np.float64)
        chi = np.zeros(count, np.float64)
        nd = np.zeros(count, np.int64)
        with np.errstate(invalid='ignore', divide='ignore', over='ignore'):
            inv_mm = 1.0 / (m64 * m64)                    # c / m^2 with c = 1
            for i in range(n_ops):
                s_i = surv[i]
                np.multiply(inv_mm, mvar[i], out=tmp)
                np.add(accv, tmp, out=accv, where=s_i)
                # chi^2 over the distinct survivors with a positive variance
                pos = s_i & (mvar[i] > 0)
                np.subtract(vals[i], mean, out=tmp)
                np.multiply(tmp, tmp, out=tmp)
                np.divide(tmp, mvar[i], out=tmp)
                np.add(chi, tmp, out=chi, where=pos)
                nd += pos
            del inv_mm
            if cand.size:
                # the same sums with the multiplicities c_j of the sources
                sc = surv[:, cand]
                mult, first = _multiplicity(keys_c, sc)
                mm = m64[cand] * m64[cand]
                mean_c = mean[cand]
                acc_c = np.zeros(cand.size, np.float64)
                chi_c = np.zeros(cand.size, np.float64)
                nd_c = np.zeros(cand.size, np.int64)
                for i in range(n_ops):
                    v64 = mvar[i, cand].astype(np.float64)
                    np.add(acc_c, (mult[i] / mm) * v64, out=acc_c, where=sc[i])
                    pos = first[i] & (v64 > 0)
                    dd = vals[i, cand] - mean_c
                    np.add(chi_c, (dd * dd) / v64, out=chi_c, where=pos)
                    nd_c += pos
                accv[cand] = acc_c
                chi[cand] = chi_c
                nd[cand] = nd_c
        out_var = np.where(m > 0, accv.astype(np.float32), np.float32(np.nan))
        del accv

    # diagnostics per shell
    shell = np.broadcast_to(_shell_of(plan, c[0], c[1], c[2]), shape).reshape(-1)
    has = n > 0
    ints = np.zeros((N_SHELLS, _NI), np.int64)
    flts = np.zeros((N_SHELLS, _NF), np.float64)

    def bc(w: Any) -> npt.NDArray[np.float64]:
        return np.bincount(shell, weights=w, minlength=N_SHELLS)

    ints[:, 0] = bc(has).astype(np.int64)
    ints[:, 1] = bc(n).astype(np.int64)
    if rej is not None:
        ints[:, 2] = bc(rej.sum(axis=0)).astype(np.int64)
        ints[:, _I_OUT] = bc(rej[plan.identity_index]).astype(np.int64)
    ints[:, 3] = bc(tested).astype(np.int64)
    ints[:, 4] = bc(floor_used).astype(np.int64)
    # each voxel of an orbit has the weight 1 / (number of distinct sources)
    np.maximum(n_src, 1.0, out=n_src)
    if has_var:
        ok = nd >= 2
        flts[:, 0] = bc(np.where(ok, chi / n_src, 0.0))
        flts[:, 1] = bc(np.where(ok, (nd - 1) / n_src, 0.0))
    rn = np.zeros(count, np.float64)
    rd = np.zeros(count, np.float64)
    with np.errstate(invalid='ignore'):
        for i in range(n_ops):
            np.subtract(vals[i], mean, out=tmp)
            np.abs(tmp, out=tmp)
            np.add(rn, tmp, out=rn, where=surv[i])
            np.add(rd, np.abs(vals[i]), out=rd, where=surv[i])
    flts[:, 2] = bc(rn / n_src)
    flts[:, 3] = bc(rd / n_src)
    return out, out_var, ints, flts


def _run_numpy(plan: OrbitPlan, data: npt.NDArray[np.float32],
               variance: npt.NDArray[np.float32] | None, var_mode: int,
               sigma: float | None, min_valid: int, poisson_floor: bool,
               progress_callback: Callable[[int, int], None] | None,
               chunk_voxels: int | None) -> OrbitResult:
    t0 = time.perf_counter()
    data3 = np.ascontiguousarray(data, dtype=np.float32)
    var3 = np.ascontiguousarray(variance, dtype=np.float32) if var_mode == 1 else None
    filled = weight = None
    if var_mode == 0 and np.any(plan.kind == 2):
        # map_coordinates input for the interpolated members (as in rspace3d 2.2)
        finite = np.isfinite(data3)
        filled = np.where(finite, data3, np.float32(0.0))
        weight = finite.astype(np.float32)
        del finite
    src = _Source(data3, data3.reshape(-1), var3,
                  var3.reshape(-1) if var3 is not None else None, var_mode, filled, weight)
    nvox = data3.size
    nh, nk, nl = plan.shape
    if chunk_voxels is None:
        # (n_ops, count) arrays of about 800,000 values stay in the CPU cache
        chunk_voxels = max(4096, int(_BOX_VALUES // max(plan.n_ops, 1)))
    chunk_voxels = int(chunk_voxels)
    # The collision test (`_collision_voxels`) runs once per slab. The
    # per-voxel work runs on the smaller boxes of each slab.
    slabs = [(s, list(_boxes_in(s, nk, nl, chunk_voxels)))
             for s in _boxes(plan.shape, max(chunk_voxels, _SLAB_VOXELS))]
    n_boxes = sum(len(b) for _, b in slabs)
    out = np.empty(nvox, np.float32)
    out_var = np.empty(nvox, np.float32) if var_mode else None
    ints = np.zeros((N_SHELLS, _NI), np.int64)
    flts = np.zeros((N_SHELLS, _NF), np.float64)
    bi = 0
    for (sh0, sh1, sk0, sk1), boxes in slabs:
        slab_shape = (sh1 - sh0, sk1 - sk0, nl)
        sources = _sources(plan, *_ranges(sh0, sh1, sk0, sk1, nl))
        cand = _collision_voxels(sources, slab_shape)
        keys = _keys_at(sources, slab_shape, cand, plan.shape) if cand.size else None
        del sources
        slab_base = (sh0 * nk + sk0) * nl
        for h0, h1, k0, k1 in boxes:
            if progress_callback:
                progress_callback(bi, n_boxes)
            bi += 1
            base = (h0 * nk + k0) * nl
            count = (h1 - h0) * (k1 - k0) * nl
            # the slab and its boxes are contiguous ranges of the flat volume
            lo, hi = np.searchsorted(cand, [base - slab_base, base - slab_base + count])
            box_keys = keys[:, lo:hi] if keys is not None else None
            o, ov, ii, ff = _box_cpu(plan, src, h0, h1, k0, k1, sigma, min_valid,
                                     poisson_floor, cand[lo:hi] - (base - slab_base), box_keys)
            out[base:base + count] = o
            if out_var is not None:
                out_var[base:base + count] = ov
            ints += ii
            flts += ff
    if progress_callback:
        progress_callback(n_boxes, n_boxes)
    return OrbitResult(
        intensity=out.reshape(plan.shape),
        variance=out_var.reshape(plan.shape) if out_var is not None else None,
        n_outliers_removed=int(ints[:, _I_OUT].sum()), int_sums=ints, float_sums=flts,
        engine='cpu-numpy', timing={'total_s': time.perf_counter() - t0})


# ──────────────────────────────────────────────────────────────────
# CPU engine (numba, optional)
# ──────────────────────────────────────────────────────────────────

NUMBA_BLOCK = 4096          # voxels per numba work item
NUMBA_SLAB = 1 << 24        # voxels per numba call (one progress step)


def numba_available() -> bool:
    """True if numba can be imported (the numba CPU engine can be used)."""
    try:
        import numba  # noqa: F401
    except ImportError:
        return False
    return True


def _run_numba(plan: OrbitPlan, data: npt.NDArray[np.float32],
               variance: npt.NDArray[np.float32] | None, var_mode: int,
               sigma: float | None, min_valid: int, poisson_floor: bool,
               progress_callback: Callable[[int, int], None] | None,
               slab_voxels: int = NUMBA_SLAB) -> OrbitResult:
    from . import _orbit_numba
    t0 = time.perf_counter()
    flat = np.ascontiguousarray(data, dtype=np.float32).reshape(-1)
    var = (np.ascontiguousarray(variance, dtype=np.float32).reshape(-1) if var_mode == 1
           else np.zeros(1, np.float32))
    nvox = flat.size
    nh, nk, nl = plan.shape
    out = np.empty(nvox, np.float32)
    out_var = np.empty(nvox if var_mode else 1, np.float32)
    ints = np.zeros((N_SHELLS, _NI), np.int64)
    flts = np.zeros((N_SHELLS, _NF), np.float64)
    slab = max(1, min(int(slab_voxels), nvox))
    n_slabs = -(-nvox // slab)
    for si, base in enumerate(range(0, nvox, slab)):
        if progress_callback:
            progress_callback(si, n_slabs)
        count = min(slab, nvox - base)
        nblocks = -(-count // NUMBA_BLOCK)
        iacc = np.zeros((nblocks, N_SHELLS, _NI), np.int64)
        facc = np.zeros((nblocks, N_SHELLS, _NF), np.float64)
        _orbit_numba.orbit_blocks(
            flat, var, var_mode, out[base:base + count],
            out_var[base:base + count] if var_mode else out_var,
            plan.tab, plan.opi, plan.opl, plan.opd, plan.geo,
            nh, nk, nl, plan.n_ops, plan.identity_index, sigma is not None,
            np.float32(sigma if sigma is not None else 0.0), int(min_valid),
            bool(poisson_floor), float(plan.edge_tol), np.float32(MAD_SCALE), _DEN_MIN,
            base, count, NUMBA_BLOCK, iacc, facc)
        ints += iacc.sum(axis=0)
        flts += facc.sum(axis=0)
    if progress_callback:
        progress_callback(n_slabs, n_slabs)
    return OrbitResult(
        intensity=out.reshape(plan.shape),
        variance=out_var.reshape(plan.shape) if var_mode else None,
        n_outliers_removed=int(ints[:, _I_OUT].sum()), int_sums=ints, float_sums=flts,
        engine='cpu-numba', timing={'total_s': time.perf_counter() - t0})


def run_cpu(plan: OrbitPlan, data: npt.NDArray[np.float32],
            variance: npt.NDArray[np.float32] | None, sigma: float | None,
            min_valid: int, poisson_floor: bool,
            progress_callback: Callable[[int, int], None] | None = None,
            chunk_voxels: int | None = None, derive_variance: bool = False,
            backend: str = 'auto') -> OrbitResult:
    """CPU engine. `data` and `variance` are float32 arrays of `plan.shape`.

    `derive_variance=True` (with `variance=None`) uses the variance max(x, 1)
    of each data value x. `backend` is 'numba', 'numpy' or 'auto'. 'auto'
    uses numba if it is installed. If the numba engine fails, 'auto' gives a
    warning and uses the NumPy engine. All backends give the same bits.
    """
    if backend not in ('auto', 'numba', 'numpy'):
        raise ValueError(f"backend must be 'auto', 'numba' or 'numpy', got {backend!r}")
    var_mode = 1 if variance is not None else 2 if derive_variance else 0
    if backend == 'numba' or (backend == 'auto' and numba_available()):
        try:
            return _run_numba(plan, data, variance, var_mode, sigma, min_valid,
                              poisson_floor, progress_callback)
        except Exception as exc:          # numba is optional: any failure falls back
            if backend == 'numba':
                raise
            warnings.warn(f"The numba CPU engine failed ({exc!r}). symmetrize_volume uses "
                          "the NumPy CPU engine (slower, same result).", RuntimeWarning,
                          stacklevel=3)
    return _run_numpy(plan, data, variance, var_mode, sigma, min_valid, poisson_floor,
                      progress_callback, chunk_voxels)


# ──────────────────────────────────────────────────────────────────
# GPU engine (CUDA through cupy.RawKernel)
# ──────────────────────────────────────────────────────────────────

_CUDA_SOURCE = r'''
#define MAXOPS 48
#define NSH 10
#define NI 6
#define NF 4
#define I_OUT 5

__device__ __forceinline__ bool fin(float x) {
    return (__float_as_uint(x) & 0x7fffffffu) < 0x7f800000u;
}

// atomicAdd(double*, double) needs compute capability 6.0 or higher.
// Older devices use a compare-and-swap loop.
__device__ __forceinline__ void add_f64(double* a, double v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
    unsigned long long* p = (unsigned long long*)a;
    unsigned long long old = *p;
    unsigned long long assumed;
    do {
        assumed = old;
        old = atomicCAS(p, assumed, (unsigned long long)__double_as_longlong(
                            __dadd_rn(__longlong_as_double((long long)assumed), v)));
    } while (assumed != old);
#else
    atomicAdd(a, v);
#endif
}

__device__ void isort(float* a, int n) {
    for (int i = 1; i < n; i++) {
        float x = a[i];
        int j = i;
        while (j > 0 && a[j - 1] > x) { a[j] = a[j - 1]; j--; }
        a[j] = x;
    }
}

__device__ float med_sorted(const float* a, int n) {
    int h = n >> 1;
    if (n & 1) return a[h];
    return __fmul_rn(__fadd_rn(a[h - 1], a[h]), 0.5f);
}

extern "C" __global__ void orbit_kernel(
    const float* __restrict__ vol, const float* __restrict__ var,
    float* __restrict__ out, float* __restrict__ out_var,
    const long long* __restrict__ tab, const int* __restrict__ opi,
    const long long* __restrict__ opl, const double* __restrict__ opd,
    const double* __restrict__ geo,
    unsigned long long* __restrict__ iacc, double* __restrict__ facc,
    long long nh, long long nk, long long nl, int n_ops, int id_op,
    int do_reject, float sigma, int min_valid, int floor_on, int has_var,
    double edge_tol, long long base, long long count)
{
    __shared__ unsigned long long s_i[NSH * NI];
    __shared__ double s_f[NSH * NF];
    for (int i = threadIdx.x; i < NSH * NI; i += blockDim.x) s_i[i] = 0ULL;
    for (int i = threadIdx.x; i < NSH * NF; i += blockDim.x) s_f[i] = 0.0;
    __syncthreads();

    long long li = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (li < count) {
        long long v = base + li;
        long long nkl = nk * nl;
        long long c0 = v / nkl;
        long long r = v - c0 * nkl;
        long long c1 = r / nl;
        long long c2 = r - c1 * nl;
        long long cc[3] = {c0, c1, c2};
        long long nn[3] = {nh, nk, nl};
        double dc0 = (double)c0, dc1 = (double)c1, dc2 = (double)c2;

        float vals[MAXOPS];
        float mvar[MAXOPS];
        long long key[MAXOPS];
        int n = 0;
        int self = -1;
        for (int o = 0; o < n_ops; o++) {
            int kind = opi[o * 8];
            float x;
            float xv = 0.0f;
            long long k;
            if (kind != 2) {
                long long src[3];
                bool ok = true;
                for (int d = 0; d < 3; d++) {
                    long long s;
                    if (kind == 0) {
                        s = tab[opi[o * 8 + 4 + d] + cc[opi[o * 8 + 1 + d]]];
                    } else {
                        const long long* m = opl + o * 12;
                        s = m[3 * d] * c0 + m[3 * d + 1] * c1 + m[3 * d + 2] * c2 + m[9 + d];
                    }
                    if (s < 0 || s >= nn[d]) { ok = false; break; }
                    src[d] = s;
                }
                if (!ok) continue;
                k = (src[0] * nk + src[1]) * nl + src[2];
                x = vol[k];
                if (!fin(x)) continue;
                if (has_var == 1) xv = var[k];
                else if (has_var == 2) xv = fmaxf(x, 1.0f);
            } else {
                const double* m = opd + o * 12;
                long long ib[3];
                double w0[3], w1[3];
                bool inside = true;
                for (int d = 0; d < 3; d++) {
                    double c = __dadd_rn(__dadd_rn(__dadd_rn(__dmul_rn(m[3 * d], dc0),
                                                             __dmul_rn(m[3 * d + 1], dc1)),
                                                   __dmul_rn(m[3 * d + 2], dc2)),
                                         m[9 + d]);
                    double hi = (double)(nn[d] - 1);
                    if (c < 0.0 && c > -edge_tol) c = 0.0;
                    if (c > hi && c < __dadd_rn(hi, edge_tol)) c = hi;
                    if (!(c >= 0.0 && c <= hi)) { inside = false; break; }
                    double f = floor(c);
                    double a0 = __dsub_rn(1.0, __dsub_rn(c, f));
                    w0[d] = a0;
                    w1[d] = __dsub_rn(1.0, a0);
                    ib[d] = (long long)f;
                }
                if (!inside) continue;
                double num = 0.0, den = 0.0;
                for (int s0 = 0; s0 < 2; s0++) {
                    double a0 = s0 ? w1[0] : w0[0];
                    if (a0 == 0.0) continue;
                    for (int s1 = 0; s1 < 2; s1++) {
                        double a1 = s1 ? w1[1] : w0[1];
                        if (a1 == 0.0) continue;
                        for (int s2 = 0; s2 < 2; s2++) {
                            double a2 = s2 ? w1[2] : w0[2];
                            if (a2 == 0.0) continue;
                            long long idx = ((ib[0] + s0) * nk + (ib[1] + s1)) * nl + ib[2] + s2;
                            float xx = vol[idx];
                            if (!fin(xx)) continue;
                            num = __dadd_rn(num, __dmul_rn(__dmul_rn(__dmul_rn((double)xx, a0), a1), a2));
                            den = __dadd_rn(den, __dmul_rn(__dmul_rn(a0, a1), a2));
                        }
                    }
                }
                float num32 = __double2float_rn(num);
                float den32 = __double2float_rn(den);
                if (!(den32 > 1e-6f)) continue;
                x = __fdiv_rn(num32, fmaxf(den32, 1e-6f));
                if (!fin(x)) continue;
                if (has_var) {
                    double vs = 0.0;
                    for (int s0 = 0; s0 < 2; s0++) {
                        double a0 = s0 ? w1[0] : w0[0];
                        if (a0 == 0.0) continue;
                        for (int s1 = 0; s1 < 2; s1++) {
                            double a1 = s1 ? w1[1] : w0[1];
                            if (a1 == 0.0) continue;
                            for (int s2 = 0; s2 < 2; s2++) {
                                double a2 = s2 ? w1[2] : w0[2];
                                if (a2 == 0.0) continue;
                                long long idx = ((ib[0] + s0) * nk + (ib[1] + s1)) * nl + ib[2] + s2;
                                float xk = vol[idx];
                                if (!fin(xk)) continue;
                                double rr = __ddiv_rn(__dmul_rn(__dmul_rn(a0, a1), a2), den);
                                double vk = has_var == 1 ? (double)var[idx] : (double)fmaxf(xk, 1.0f);
                                vs = __dadd_rn(vs, __dmul_rn(__dmul_rn(rr, rr), vk));
                            }
                        }
                    }
                    xv = __double2float_rn(vs);
                }
                if (w1[0] == 0.0 && w1[1] == 0.0 && w1[2] == 0.0)
                    k = (ib[0] * nk + ib[1]) * nl + ib[2];
                else
                    k = -1 - (long long)o;
            }
            if (o == id_op) self = n;
            vals[n] = x;
            mvar[n] = xv;
            key[n] = k;
            n++;
        }

        // median, MAD and rejection
        float med = 0.0f;
        float thr = 0.0f;
        bool tested = false, floor_used = false;
        if (do_reject && n >= min_valid && n > 0) {
            float s[MAXOPS];
            for (int i = 0; i < n; i++) s[i] = vals[i];
            isort(s, n);
            med = med_sorted(s, n);
            for (int i = 0; i < n; i++) s[i] = fabsf(__fsub_rn(vals[i], med));
            isort(s, n);
            float scale = __fmul_rn(med_sorted(s, n), 1.4826f);
            if (floor_on) {
                float root = __fsqrt_rn(fmaxf(med, 1.0f));
                floor_used = root > scale;
                scale = fmaxf(scale, root);
            }
            thr = __fmul_rn(sigma, scale);
            tested = true;
        }
        unsigned long long surv = 0ULL;
        float acc = 0.0f;
        int m = 0;
        int n_rej = 0;
        bool self_rej = false;
        for (int i = 0; i < n; i++) {
            if (tested && fabsf(__fsub_rn(vals[i], med)) > thr) {
                n_rej++;
                if (i == self) self_rej = true;
            } else {
                surv |= 1ULL << i;
                acc = __fadd_rn(acc, vals[i]);
                m++;
            }
        }
        float res = m > 0 ? __fdiv_rn(acc, (float)m) : __int_as_float(0x7fc00000);
        out[li] = res;

        // rep[i]: the first finite member that reads the same source as member i
        unsigned char rep[MAXOPS];
        int n_src = 0;
        for (int i = 0; i < n; i++) {
            int r0 = i;
            for (int j = 0; j < i; j++) {
                if (key[j] == key[i]) { r0 = j; break; }
            }
            rep[i] = (unsigned char)r0;
            if (r0 == i) n_src++;
        }

        double chi = 0.0;
        int nd = 0;
        if (has_var) {
            // cnt[g]: number of survivors that read the source of group g
            unsigned char cnt[MAXOPS];
            for (int i = 0; i < n; i++) cnt[i] = 0;
            for (int i = 0; i < n; i++)
                if ((surv >> i) & 1ULL) cnt[rep[i]]++;
            unsigned long long seen = 0ULL;
            double accv = 0.0;
            double mm = __dmul_rn((double)m, (double)m);
            double mean = (double)res;
            for (int i = 0; i < n; i++) {
                if (!((surv >> i) & 1ULL)) continue;
                int g = rep[i];
                bool first = !((seen >> g) & 1ULL);
                seen |= 1ULL << g;
                double f = __ddiv_rn((double)cnt[g], mm);
                accv = __dadd_rn(accv, __dmul_rn(f, (double)mvar[i]));
                if (first && mvar[i] > 0.0f) {
                    double dd = __dsub_rn((double)vals[i], mean);
                    chi = __dadd_rn(chi, __ddiv_rn(__dmul_rn(dd, dd), (double)mvar[i]));
                    nd++;
                }
            }
            out_var[li] = m > 0 ? __double2float_rn(accv) : __int_as_float(0x7fc00000);
        }

        if (n > 0) {
            double h[3], g[3];
            for (int d = 0; d < 3; d++)
                h[d] = __dadd_rn(geo[d], __dadd_rn(__dadd_rn(__dmul_rn(geo[3 + 3 * d], dc0),
                                                             __dmul_rn(geo[4 + 3 * d], dc1)),
                                                   __dmul_rn(geo[5 + 3 * d], dc2)));
            for (int d = 0; d < 3; d++)
                g[d] = __dadd_rn(__dadd_rn(__dmul_rn(geo[12 + 3 * d], h[0]),
                                           __dmul_rn(geo[13 + 3 * d], h[1])),
                                 __dmul_rn(geo[14 + 3 * d], h[2]));
            double q2 = __dadd_rn(__dadd_rn(__dmul_rn(h[0], g[0]), __dmul_rn(h[1], g[1])),
                                  __dmul_rn(h[2], g[2]));
            double q = __dsqrt_rn(fmax(q2, 0.0));
            int sh = (int)__dmul_rn(q, geo[21]);
            if (sh > NSH - 1) sh = NSH - 1;
            if (sh < 0) sh = 0;
            unsigned long long* si = s_i + sh * NI;
            atomicAdd(si + 0, 1ULL);
            atomicAdd(si + 1, (unsigned long long)n);
            if (n_rej) atomicAdd(si + 2, (unsigned long long)n_rej);
            if (tested) atomicAdd(si + 3, 1ULL);
            if (floor_used) atomicAdd(si + 4, 1ULL);
            if (self_rej) atomicAdd(si + I_OUT, 1ULL);
            // the float sums have the weight 1 / (number of distinct sources)
            double ds = (double)n_src;
            if (has_var && nd >= 2) {
                add_f64(s_f + sh * NF + 0, __ddiv_rn(chi, ds));
                add_f64(s_f + sh * NF + 1, __ddiv_rn((double)(nd - 1), ds));
            }
            if (m > 0) {
                double mean = (double)res;
                double rn = 0.0, rd = 0.0;
                for (int i = 0; i < n; i++) {
                    if (!((surv >> i) & 1ULL)) continue;
                    rn += fabs((double)vals[i] - mean);
                    rd += fabs((double)vals[i]);
                }
                add_f64(s_f + sh * NF + 2, __ddiv_rn(rn, ds));
                add_f64(s_f + sh * NF + 3, __ddiv_rn(rd, ds));
            }
        }
    }
    __syncthreads();
    for (int i = threadIdx.x; i < NSH * NI; i += blockDim.x)
        if (s_i[i]) atomicAdd(iacc + i, s_i[i]);
    for (int i = threadIdx.x; i < NSH * NF; i += blockDim.x)
        if (s_f[i] != 0.0) add_f64(facc + i, s_f[i]);
}
'''

_KERNEL = None
SLAB_VOXELS = 1 << 24          # output voxels per kernel launch
_MIN_SLAB = 1 << 16
_UPLOAD_BYTES = 256 * 2**20    # largest host-to-GPU copy


def _kernel() -> Any:
    """The compiled CUDA kernel. A compilation error is raised here, not at the launch."""
    global _KERNEL
    if _KERNEL is None:
        import cupy as cp
        kern = cp.RawKernel(_CUDA_SOURCE, 'orbit_kernel', options=('--fmad=false',))
        kern.compile()
        _KERNEL = kern
    return _KERNEL


def _gpu_errors() -> tuple[type[BaseException], ...]:
    """CuPy exceptions of a failed compilation, upload or launch."""
    import cupy as cp
    from cupy.cuda import compiler
    errors: list[type[BaseException]] = [
        compiler.CompileException, cp.cuda.runtime.CUDARuntimeError,
        cp.cuda.driver.CUDADriverError, cp.cuda.memory.OutOfMemoryError]
    try:
        from cupy_backends.cuda.libs.nvrtc import NVRTCError
        errors.append(NVRTCError)
    except ImportError:
        pass
    return tuple(errors)


def _free_gpu_memory() -> None:
    """Release the blocks of the CuPy memory pool. Errors are ignored."""
    try:
        import cupy as cp
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:          # noqa: BLE001 - the device can be in an error state
        pass


def _upload(cp: Any, host: npt.NDArray[np.float32]) -> Any:
    """Copy a C-contiguous float32 array to the GPU in slices of at most 256 MB."""
    flat = host.reshape(-1)
    dev = cp.empty(flat.size, dtype=cp.float32)
    step = max(1, _UPLOAD_BYTES // 4)
    for i in range(0, flat.size, step):
        dev[i:i + step].set(flat[i:i + step])
    return dev


def run_gpu(plan: OrbitPlan, data: npt.NDArray[np.float32],
            variance: npt.NDArray[np.float32] | None, sigma: float | None,
            min_valid: int, poisson_floor: bool,
            progress_callback: Callable[[int, int], None] | None = None,
            slab_voxels: int = SLAB_VOXELS, derive_variance: bool = False) -> OrbitResult | None:
    """CUDA engine. Returns None if the input volume does not fit in GPU memory.

    `derive_variance=True` (with `variance=None`) uses the variance max(x, 1)
    of each data value x. The kernel calculates it from the data value, so
    that no variance array is copied to the GPU.
    """
    import cupy as cp
    oom = (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError)
    t0 = time.perf_counter()
    kern = _kernel()
    host = np.ascontiguousarray(data, dtype=np.float32)
    host_var = (np.ascontiguousarray(variance, dtype=np.float32)
                if variance is not None else None)
    var_mode = 1 if host_var is not None else 2 if derive_variance else 0
    nvox = host.size
    pool = cp.get_default_memory_pool()
    d_vol = d_var = None
    try:
        d_vol = _upload(cp, host)
        d_var = _upload(cp, host_var) if host_var is not None else cp.zeros(1, cp.float32)
    except oom:
        del d_vol, d_var
        pool.free_all_blocks()
        return None
    t_upload = time.perf_counter() - t0

    d_tab = cp.asarray(plan.tab)
    d_opi = cp.asarray(np.ascontiguousarray(plan.opi))
    d_opl = cp.asarray(np.ascontiguousarray(plan.opl))
    d_opd = cp.asarray(np.ascontiguousarray(plan.opd))
    d_geo = cp.asarray(plan.geo)
    d_iacc = cp.zeros(N_SHELLS * _NI, cp.uint64)
    d_facc = cp.zeros(N_SHELLS * _NF, cp.float64)

    slab = max(1, min(int(slab_voxels), nvox))
    d_out = d_out_var = None
    while True:
        try:
            d_out = cp.empty(slab, cp.float32)
            d_out_var = cp.empty(slab if var_mode else 1, cp.float32)
            break
        except oom:
            d_out = d_out_var = None
            pool.free_all_blocks()
            if slab <= _MIN_SLAB:
                del d_vol, d_var
                pool.free_all_blocks()
                return None
            slab = max(_MIN_SLAB, slab // 2)

    out = np.empty(nvox, np.float32)
    out_var = np.empty(nvox, np.float32) if var_mode else None
    nh, nk, nl = plan.shape
    threads = 256
    n_slabs = -(-nvox // slab)
    t_kernel = 0.0
    start, stop = cp.cuda.Event(), cp.cuda.Event()
    for si, base in enumerate(range(0, nvox, slab)):
        if progress_callback:
            progress_callback(si, n_slabs)
        count = min(slab, nvox - base)
        start.record()
        kern(((count + threads - 1) // threads,), (threads,),
             (d_vol, d_var, d_out, d_out_var, d_tab, d_opi, d_opl, d_opd, d_geo,
              d_iacc, d_facc, np.int64(nh), np.int64(nk), np.int64(nl),
              np.int32(plan.n_ops), np.int32(plan.identity_index),
              np.int32(sigma is not None),
              np.float32(sigma if sigma is not None else 0.0),
              np.int32(min_valid), np.int32(bool(poisson_floor)),
              np.int32(var_mode), np.float64(plan.edge_tol),
              np.int64(base), np.int64(count)))
        stop.record()
        stop.synchronize()
        t_kernel += cp.cuda.get_elapsed_time(start, stop) / 1000.0
        d_out[:count].get(out=out[base:base + count])
        if out_var is not None:
            d_out_var[:count].get(out=out_var[base:base + count])
    if progress_callback:
        progress_callback(n_slabs, n_slabs)
    ints = cp.asnumpy(d_iacc).astype(np.int64).reshape(N_SHELLS, _NI)
    flts = cp.asnumpy(d_facc).reshape(N_SHELLS, _NF)
    del d_vol, d_var, d_out, d_out_var, d_tab, d_opi, d_opl, d_opd, d_geo, d_iacc, d_facc
    pool.free_all_blocks()
    return OrbitResult(
        intensity=out.reshape(plan.shape),
        variance=out_var.reshape(plan.shape) if out_var is not None else None,
        n_outliers_removed=int(ints[:, _I_OUT].sum()), int_sums=ints, float_sums=flts,
        engine='gpu',
        timing={'total_s': time.perf_counter() - t0, 'upload_s': t_upload,
                'kernel_s': t_kernel, 'slab_voxels': float(slab)})


def run(plan: OrbitPlan, data: npt.NDArray[np.float32],
        variance: npt.NDArray[np.float32] | None, sigma: float | None,
        min_valid: int, poisson_floor: bool, use_gpu: bool,
        progress_callback: Callable[[int, int], None] | None = None,
        derive_variance: bool = False) -> OrbitResult:
    """Run the GPU engine if `use_gpu`, else the CPU engine.

    If the input does not fit in GPU memory, a warning is given and the CPU
    engine is used. If the CUDA kernel does not compile, or if a CUDA
    runtime or driver error occurs, a warning is given and the CPU engine is
    used. `derive_variance` is explained in `run_cpu`.
    """
    if use_gpu:
        failure = None
        try:
            res = run_gpu(plan, data, variance, sigma, min_valid, poisson_floor,
                          progress_callback, derive_variance=derive_variance)
        except Exception as exc:
            if not isinstance(exc, _gpu_errors()):
                raise
            failure = repr(exc)
            res = None
        if failure is not None:
            # the GPU arrays of run_gpu are released when the exception is deleted
            _free_gpu_memory()
            warnings.warn(f"The GPU engine failed ({failure}). symmetrize_volume uses the "
                          "CPU engine (slower, same result).", RuntimeWarning, stacklevel=3)
        elif res is not None:
            return res
        else:
            warnings.warn("The volume does not fit in GPU memory. symmetrize_volume uses "
                          "the CPU engine (slower, same result).", RuntimeWarning,
                          stacklevel=3)
    return run_cpu(plan, data, variance, sigma, min_valid, poisson_floor, progress_callback,
                   derive_variance=derive_variance)


# ──────────────────────────────────────────────────────────────────
# Diagnostics
# ──────────────────────────────────────────────────────────────────

def _ratio(a: float, b: float) -> float | None:
    return float(a) / float(b) if b else None


def diagnostics_from_sums(shell_edges: list[float], q_unit: str, has_variance: bool,
                          int_sums: Any, float_sums: Any) -> dict[str, Any]:
    """Diagnostics dict (lists, floats, None) from the per-shell sums.

    `int_sums` is (N_SHELLS, 6) and `float_sums` (N_SHELLS, 4), columns as in
    `_INT_KEYS` and `_FLOAT_KEYS`. The ratios are None where they are not
    defined. Without variance, chi2 and nu are None.

    chi2, nu, rsym_num and rsym_den are weighted sums. Each voxel adds its
    value divided by the number of distinct sources among its finite orbit
    members, so that an exact orbit counts once. With interpolated members
    this weight is approximate. The voxel, member, rejected and floor counts
    are not weighted.
    """
    ints = np.asarray(int_sums, np.int64).reshape(-1, _NI)
    flts = np.asarray(float_sums, np.float64).reshape(-1, _NF)

    def block(ii: Any, ff: Any) -> dict[str, Any]:
        d: dict[str, Any] = {k: [int(x) for x in ii[:, c]] for c, k in enumerate(_INT_KEYS)}
        d.update({k: [float(x) for x in ff[:, c]] for c, k in enumerate(_FLOAT_KEYS)})
        rows = range(ii.shape[0])
        d['rejected_fraction'] = [_ratio(ii[r, 2], ii[r, 1]) for r in rows]
        d['poisson_floor_fraction'] = [_ratio(ii[r, 4], ii[r, 3]) for r in rows]
        d['r_sym'] = [_ratio(ff[r, 2], ff[r, 3]) for r in rows]
        if has_variance:
            d['chi2_per_nu'] = [_ratio(ff[r, 0], ff[r, 1]) for r in rows]
        else:
            d['chi2'] = [None] * len(rows)
            d['nu'] = [None] * len(rows)
            d['chi2_per_nu'] = [None] * len(rows)
        return d

    shells = block(ints, flts)
    total = {k: v[0] for k, v in block(ints.sum(axis=0, keepdims=True),
                                       flts.sum(axis=0, keepdims=True)).items()}
    return {'q_unit': q_unit, 'n_shells': int(ints.shape[0]),
            'shell_edges': [float(x) for x in shell_edges],
            'has_variance': bool(has_variance), 'shells': shells, 'total': total}


def diagnostics_to_arrays(diag: dict[str, Any]) -> dict[str, npt.NDArray[Any]]:
    """Per-shell sums of a diagnostics dict as named 1D arrays (for HDF5).

    chi2 and nu are 0 when the volume has no variance.
    """
    sh = diag['shells']
    arrays: dict[str, npt.NDArray[Any]] = {
        'shell_edges': np.asarray(diag['shell_edges'], np.float64)}
    for k in _INT_KEYS:
        arrays[k] = np.asarray([x or 0 for x in sh[k]], np.int64)
    for k in _FLOAT_KEYS:
        arrays[k] = np.asarray([x or 0.0 for x in sh[k]], np.float64)
    return arrays


def diagnostics_from_arrays(arrays: dict[str, Any], q_unit: str,
                            has_variance: bool) -> dict[str, Any]:
    """Inverse of `diagnostics_to_arrays`."""
    ints = np.stack([np.asarray(arrays[k], np.int64) for k in _INT_KEYS], axis=1)
    flts = np.stack([np.asarray(arrays[k], np.float64) for k in _FLOAT_KEYS], axis=1)
    return diagnostics_from_sums(list(np.asarray(arrays['shell_edges'], np.float64)),
                                 q_unit, has_variance, ints, flts)


def result_diagnostics(plan: OrbitPlan, res: OrbitResult) -> dict[str, Any]:
    edges = [plan.q_max * i / N_SHELLS for i in range(N_SHELLS + 1)]
    return diagnostics_from_sums(edges, plan.q_unit, res.variance is not None,
                                 res.int_sums, res.float_sums)


def format_symmetry_diagnostics(diag: dict[str, Any] | None) -> str:
    """Short text table of `metadata['symmetry_diagnostics']`."""
    if not diag:
        return 'no symmetry diagnostics'
    sh = diag['shells']
    edges = diag['shell_edges']
    unit = diag.get('q_unit', '')

    def fmt(x: float | None, spec: str) -> str:
        return format(x, spec) if x is not None else '-'

    lines = [f"|q| shell ({unit})      voxels   members  rejected  floor    R_sym  chi2/nu"]
    for i in range(len(edges) - 1):
        lines.append(
            f"{edges[i]:7.4f} - {edges[i + 1]:7.4f} {sh['n_voxels'][i]:10d} {sh['n_members'][i]:9d} "
            f"{fmt(sh['rejected_fraction'][i] and 100 * sh['rejected_fraction'][i], '8.3f')}% "
            f"{fmt(sh['poisson_floor_fraction'][i] and 100 * sh['poisson_floor_fraction'][i], '5.1f')}% "
            f"{fmt(sh['r_sym'][i], '8.4f')} {fmt(sh['chi2_per_nu'][i], '8.3f')}")
    t = diag['total']
    lines.append(
        f"{'all':>17s} {t['n_voxels']:10d} {t['n_members']:9d} "
        f"{fmt(t['rejected_fraction'] and 100 * t['rejected_fraction'], '8.3f')}% "
        f"{fmt(t['poisson_floor_fraction'] and 100 * t['poisson_floor_fraction'], '5.1f')}% "
        f"{fmt(t['r_sym'], '8.4f')} {fmt(t['chi2_per_nu'], '8.3f')}")
    lines.append("rejected = rejected orbit members / finite orbit members.")
    lines.append("floor = voxels with the scale set by the Poisson floor / voxels tested.")
    return '\n'.join(lines)
