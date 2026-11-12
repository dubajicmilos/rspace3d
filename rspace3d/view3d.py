"""
view3d.py - geometry, camera and data statistics of the 3D view.

This module uses NumPy only. The OpenGL part is in view3d_gl.py and the window
is in view3d_window.py.

Coordinate systems
    index    (ih, ik, il): position in the array `intensity[ih, ik, il]`
    texture  (s, t, r), each 0..1: s along l, t along k, r along h. This is the
             order of the array in memory. A voxel centre is at (i + 0.5) / n.
    Miller   (h, k, l) = origin + A @ index (`volume_affine`; it has the shear
             of an unwarp raster)
    world    Cartesian reciprocal space in 1/A: B @ (h, k, l), with the B matrix
             of the unit cell. A volume without a cell uses B = 1.

`first_hit` and `ray_max` do on the CPU what the shader does on the GPU. The
window uses them for the read-out below the mouse, and the tests use them as
the reference for the shader.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt

from .volume_builder import VolumeData, volume_affine

BLOCK = 8                               # voxels along one side of a block of the skip texture
MAX_TEXTURE_VOXELS = 512 * 2**20        # 2 GiB of float32, about 812^3 voxels
TAN_HALF_FOV = math.tan(math.radians(12.0))
LEVEL_COLORS = ((0.12, 0.37, 0.66), (0.90, 0.54, 0.12), (0.16, 0.60, 0.50))
MAX_LEVELS = 3
AXIS_NAMES = ('h', 'k', 'l')
_TEX_OF_AXIS = (2, 1, 0)                # h is the texture axis r, k is t, l is s

Array = npt.NDArray[np.float64]


# ──────────────────────────────────────────────────────────────────
# Cell and grid
# ──────────────────────────────────────────────────────────────────

def b_matrix(cell: dict[str, float] | None) -> Array:
    """Busing-Levy B matrix: Q = B @ (h, k, l) in 1/A. a* is along x, b* is in the xy plane.

    Returns the unit matrix if there is no cell or the cell is not possible.
    """
    if not cell:
        return np.eye(3)
    try:
        a, b, c = float(cell['a']), float(cell['b']), float(cell['c'])
        al, be, ga = np.radians([float(cell['alpha']), float(cell['beta']), float(cell['gamma'])])
    except (KeyError, TypeError, ValueError):
        return np.eye(3)
    if not np.all(np.isfinite([a, b, c, al, be, ga])):
        return np.eye(3)
    ca, cb, cg = np.cos([al, be, ga])
    sa, sb, sg = np.sin([al, be, ga])
    v2 = 1.0 - ca * ca - cb * cb - cg * cg + 2.0 * ca * cb * cg
    if min(a, b, c) <= 0 or v2 <= 1e-12 or min(sa, sb, sg) <= 1e-9:
        return np.eye(3)
    vol = a * b * c * math.sqrt(v2)
    astar, bstar, cstar = b * c * sa / vol, a * c * sb / vol, a * b * sg / vol
    cbs = (ca * cg - cb) / (sa * sg)
    cgs = (ca * cb - cg) / (sa * sb)
    sbs, sgs = math.sqrt(max(1.0 - cbs * cbs, 0.0)), math.sqrt(max(1.0 - cgs * cgs, 0.0))
    return np.array([[astar, bstar * cgs, cstar * cbs],
                     [0.0, bstar * sgs, -cstar * sbs * ca],
                     [0.0, 0.0, 1.0 / c]])


def measured(x: npt.NDArray[Any]) -> npt.NDArray[np.bool_]:
    """True where a voxel has a value. NaN and exact zero mean "not measured"."""
    return np.isfinite(x) & (x != 0)


def reduce_factors(shape: tuple[int, ...], max_voxels: float = MAX_TEXTURE_VOXELS,
                   max_size: int = 2048) -> tuple[int, int, int]:
    """Whole factors along h, k and l that make the volume fit in one texture.

    An axis that is longer than `max_size` gets a larger factor. If the volume
    then has more than `max_voxels` voxels, the longest axis gets a larger
    factor, again and again.
    """
    f = [1, 1, 1]
    while True:
        dims = [max(n // k, 1) for n, k in zip(shape, f)]
        long = [i for i in range(3) if dims[i] > max_size]
        if long:
            for i in long:
                f[i] += 1
        elif float(np.prod(dims)) > max_voxels:
            f[int(np.argmax(dims))] += 1
        else:
            return f[0], f[1], f[2]


def reduce_volume(data: npt.NDArray[Any], factors: tuple[int, int, int]) -> npt.NDArray[np.float32]:
    """Mean of the measured voxels of each block. A block without a measured voxel is NaN."""
    f_h, f_k, f_l = (min(f, n) for f, n in zip(factors, data.shape))
    nh, nk, nl = (max(n // f, 1) for n, f in zip(data.shape, factors))
    out = np.empty((nh, nk, nl), np.float32)
    for j in range(nh):
        blk = np.asarray(data[j * f_h:(j + 1) * f_h, :nk * f_k, :nl * f_l], dtype=np.float32)
        blk = blk.reshape(f_h, nk, f_k, nl, f_l)
        ok = measured(blk)
        total = np.where(ok, blk, 0.0).sum(axis=(0, 2, 4), dtype=np.float64)
        count = ok.sum(axis=(0, 2, 4))
        out[j] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def _apply(m: Array, pts: npt.ArrayLike) -> Array:
    p = np.atleast_2d(np.asarray(pts, dtype=np.float64))
    return p @ m[:3, :3].T + m[:3, 3]


@dataclass
class Grid:
    """Geometry of the texture that shows a volume."""
    shape: tuple[int, int, int]          # voxels of the texture along h, k, l
    span: tuple[int, int, int]           # voxels of the volume that the texture covers
    factors: tuple[int, int, int]        # voxels of the volume in one voxel of the texture, per axis
    t2h: Array                           # 4x4: texture (s, t, r) -> Miller indices
    t2w: Array                           # 4x4: texture -> world
    w2t: Array                           # 4x4: world -> texture
    b: Array                             # 3x3: Miller indices -> world
    axes: tuple[Array, Array, Array]     # H, K, L of the volume
    center: Array                        # centre of the box in world coordinates
    radius: float                        # largest distance of a box corner from the centre
    has_cell: bool

    def hkl(self, tex: npt.ArrayLike) -> Array:
        return _apply(self.t2h, tex)

    def world(self, tex: npt.ArrayLike) -> Array:
        return _apply(self.t2w, tex)

    def tex(self, world: npt.ArrayLike) -> Array:
        return _apply(self.w2t, world)

    def tex_range(self, axis: int, lo: float, hi: float) -> tuple[float, float]:
        """Part 0..1 of the texture along h (0), k (1) or l (2) for the Miller range lo..hi.

        The range is that of the axis array of the volume (`vol.H`, `vol.K`, `vol.L`).
        The voxels at both ends are in the result, and it has at least one voxel.
        """
        ax, n = self.axes[axis], self.span[axis]
        step = float(ax[-1] - ax[0]) / (len(ax) - 1) if len(ax) > 1 else 1.0
        i0, i1 = (lo - float(ax[0])) / step, (hi - float(ax[0])) / step
        a = min(max(min(i0, i1) / n, 0.0), 1.0)
        b = min(max((max(i0, i1) + 1.0) / n, 0.0), 1.0)
        if b - a < 1.0 / n:
            a = min(a, 1.0 - 1.0 / n)
            b = a + 1.0 / n
        return a, b

    def box(self, h: tuple[float, float], k: tuple[float, float],
            l: tuple[float, float]) -> tuple[tuple[float, ...], tuple[float, ...]]:
        """Corners (lo, hi) of a box in texture coordinates for three Miller ranges."""
        (h0, h1), (k0, k1), (l0, l1) = (self.tex_range(0, *h), self.tex_range(1, *k),
                                        self.tex_range(2, *l))
        return (l0, k0, h0), (l1, k1, h1)


def make_grid(vol: VolumeData, max_voxels: float = MAX_TEXTURE_VOXELS,
              max_size: int = 2048) -> Grid:
    """Geometry of the texture for `vol`. A large volume gets reduction factors above 1."""
    full = tuple(int(n) for n in vol.intensity.shape)
    f = reduce_factors(full, max_voxels, max_size)
    shape = tuple(max(n // k, 1) for n, k in zip(full, f))
    span = tuple(s * min(k, n) for s, k, n in zip(shape, f, full))
    origin, a = volume_affine(vol)                       # hkl = origin + A @ (ih, ik, il)
    t2i = np.zeros((4, 4))                               # (s, t, r, 1) -> (ih, ik, il, 1)
    t2i[0, 2], t2i[1, 1], t2i[2, 0] = span
    t2i[:3, 3] = -0.5
    t2i[3, 3] = 1.0
    i2h = np.eye(4)
    i2h[:3, :3], i2h[:3, 3] = a, origin
    t2h = i2h @ t2i
    cell = vol.metadata.get('cell')
    b = b_matrix(cell)
    h2w = np.eye(4)
    h2w[:3, :3] = b
    t2w = h2w @ t2h
    corners = np.array(list(product((0.0, 1.0), repeat=3)))
    w = _apply(t2w, corners)
    center = 0.5 * (w.min(axis=0) + w.max(axis=0))
    return Grid(shape=shape, span=span, factors=f, t2h=t2h, t2w=t2w, w2t=np.linalg.inv(t2w), b=b,
                axes=(np.asarray(vol.H, float), np.asarray(vol.K, float), np.asarray(vol.L, float)),
                center=center, radius=float(np.linalg.norm(w - center, axis=1).max()),
                has_cell=not np.array_equal(b, np.eye(3)))


def block_max(data: npt.NDArray[Any], b: int = BLOCK) -> npt.NDArray[np.float32]:
    """Largest value in each b^3 block and in the 26 blocks around it.

    A ray can jump over a block if this value is below the level: no sample in
    the block, which the GPU interpolates from the block and one voxel around
    it, can then reach the level. NaN counts as no value.
    """
    from scipy.ndimage import maximum_filter
    nh, nk, nl = data.shape
    ch, ck, cl = -(-nh // b), -(-nk // b), -(-nl // b)
    out = np.zeros((ch, ck, cl), np.float32)
    pad = np.full((ck * b, cl * b), -np.inf, np.float32)
    for j in range(ch):
        pad[:nk, :nl] = np.fmax.reduce(data[j * b:(j + 1) * b], axis=0)
        out[j] = np.fmax.reduce(np.fmax.reduce(pad.reshape(ck, b, cl, b), axis=3), axis=1)
    out = np.nan_to_num(out, nan=0.0, neginf=0.0, posinf=3e38)
    return maximum_filter(out, size=3, mode='nearest')


# ──────────────────────────────────────────────────────────────────
# Statistics for the levels
# ──────────────────────────────────────────────────────────────────

def sample_values(data: npt.NDArray[Any], target: int = 2_000_000) -> Array:
    """Sorted positive values of `target` voxels at random places (all voxels of a small volume).

    A regular step through the volume is not used: the Bragg peaks are on a
    regular lattice, so a regular step can hit too many or too few of them.
    """
    if data.size <= target:
        x = np.asarray(data, dtype=np.float64).ravel()
    else:
        pick = np.random.default_rng(0).integers(0, data.size, size=target)
        pick.sort()                                      # in the order of the memory
        x = np.asarray(data.reshape(-1)[pick], dtype=np.float64)
    x = x[np.isfinite(x) & (x > 0)]
    x.sort()
    return x


def percentile(sample: Array, q: float) -> float:
    """Percentile q (0..100) of a sorted sample. 1.0 for an empty sample."""
    if sample.size == 0:
        return 1.0
    return float(sample[min(int(q / 100.0 * sample.size), sample.size - 1)])


def histogram(sample: Array, bins: int = 96) -> tuple[Array, Array]:
    """Counts of log10(value) between the 0.1th percentile and the maximum: (edges, counts)."""
    if sample.size == 0:
        return np.linspace(0.0, 1.0, bins + 1), np.zeros(bins)
    lo, hi = math.log10(percentile(sample, 0.1)), math.log10(float(sample[-1]))
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    counts, edges = np.histogram(np.log10(sample), bins=bins, range=(lo, hi))
    return edges, counts.astype(np.float64)


# ──────────────────────────────────────────────────────────────────
# Scene and camera
# ──────────────────────────────────────────────────────────────────

@dataclass
class Level:
    """One surface: the intensity, the colour (r, g, b in 0..1) and the opacity (0..1)."""
    value: float
    color: tuple[float, float, float] = LEVEL_COLORS[0]
    opacity: float = 1.0


@dataclass
class Scene:
    """What the renderer draws."""
    mode: str = 'surface'                # 'surface' or 'projection'
    levels: list[Level] = field(default_factory=list)
    lo: tuple[float, ...] = (0.0, 0.0, 0.0)      # shown box in texture coordinates (s, t, r)
    hi: tuple[float, ...] = (1.0, 1.0, 1.0)
    clim: tuple[float, float] = (1.0, 1.0e4)     # intensity range of the colour map (projection)
    colormap: str = 'inferno'
    dark: bool = False
    grid: bool = True
    grid_step: tuple[float, ...] = (1.0, 1.0, 1.0)
    step: float = 1.0                    # step of a ray in voxels
    skip: bool = True                    # use the block texture
    jitter: float = 1.0                  # random start of each ray (projection), 0 = off
    floor: float = 1.0e-3                # smallest intensity that has a logarithm
    debug: int = 0                       # 1: colour = distance of the first hit, 2: colour = maximum

    def sorted_levels(self) -> list[Level]:
        return sorted(self.levels, key=lambda lv: lv.value)[:MAX_LEVELS]


def new_level(sample: Array, levels: list[float]) -> float:
    """A useful intensity for one more surface.

    It is the first of the levels that 10 %, 0.1 % and 1 % of the voxels with
    a positive intensity are above which is not near a level that there is
    already (a factor of 1.5). If all three are near, it is half of the lowest
    level.
    """
    for q in (90.0, 99.9, 99.0):
        value = percentile(sample, q)
        if all(max(value, v) / max(min(value, v), 1.0e-300) > 1.5 for v in levels):
            return value
    return 0.5 * min(levels) if levels else percentile(sample, 99.0)


def default_scene(sample: Array) -> Scene:
    """One surface at the level that 1 % of the voxels with a positive intensity are above."""
    p1, p50, top = percentile(sample, 1), percentile(sample, 50), percentile(sample, 99.99)
    if top <= p50:
        top = p50 * 10.0
    # Just above the percentile: for data in steps (counts), the many voxels that are
    # equal to the percentile are then outside the surface. But not above the largest value.
    level = percentile(sample, 99)
    if sample.size and level * 1.0001 <= float(sample[-1]):
        level *= 1.0001
    return Scene(levels=[Level(level)], clim=(p50, top), floor=max(p1 * 1.0e-3, 1.0e-12))


@dataclass
class Camera:
    """A camera that turns about the vertical axis of the world (z) around a target point."""
    yaw: float = math.radians(-55.0)     # direction to the camera in the xy plane
    pitch: float = math.radians(22.0)    # height of the camera above the xy plane
    zoom: float = 1.0
    target: Array = field(default_factory=lambda: np.zeros(3))   # offset from the box centre
    ortho: bool = True

    LIMIT = math.pi / 2 - 1.0e-4

    def basis(self) -> tuple[Array, Array, Array]:
        """Unit vectors (forward, right, up) of the camera in world coordinates."""
        cp = math.cos(self.pitch)
        out = np.array([cp * math.cos(self.yaw), cp * math.sin(self.yaw), math.sin(self.pitch)])
        right = np.array([-math.sin(self.yaw), math.cos(self.yaw), 0.0])
        return -out, right, np.cross(right, -out)

    def half_height(self, grid: Grid) -> float:
        """Half of the height of the view at the target, in world units."""
        return 1.02 * grid.radius / self.zoom

    def eye(self, grid: Grid) -> Array:
        fwd = self.basis()[0]
        point = grid.center + self.target
        if self.ortho:
            return point - fwd * (3.0 * grid.radius + float(np.linalg.norm(self.target)))
        return point - fwd * (self.half_height(grid) / TAN_HALF_FOV)

    def ray(self, grid: Grid, x: float, y: float, width: float, height: float) -> tuple[Array, Array]:
        """Origin and unit direction of the view ray through the pixel (x, y), y down."""
        fwd, right, up = self.basis()
        nx = (2.0 * x / width - 1.0) * (width / height)
        ny = 1.0 - 2.0 * y / height
        eye = self.eye(grid)
        if self.ortho:
            return eye + (nx * right + ny * up) * self.half_height(grid), fwd
        d = fwd + (nx * right + ny * up) * TAN_HALF_FOV
        return eye, d / np.linalg.norm(d)

    def project(self, grid: Grid, world: npt.ArrayLike, width: float, height: float) -> Array:
        """Pixel positions (x, y), y down, of points in world coordinates."""
        fwd, right, up = self.basis()
        rel = np.atleast_2d(np.asarray(world, dtype=np.float64)) - self.eye(grid)
        if self.ortho:
            scale = np.full(len(rel), self.half_height(grid))
        else:
            scale = np.maximum(rel @ fwd, 1.0e-9) * TAN_HALF_FOV
        nx, ny = (rel @ right) / scale, (rel @ up) / scale
        return np.stack([(nx * (height / width) + 1.0) * 0.5 * width, (1.0 - ny) * 0.5 * height], axis=1)

    def orbit(self, dx: float, dy: float) -> None:
        """Turn by a mouse movement of (dx, dy) pixels."""
        self.yaw -= 0.008 * dx
        self.pitch = min(max(self.pitch + 0.008 * dy, -self.LIMIT), self.LIMIT)

    def pan(self, grid: Grid, dx: float, dy: float, height: float) -> None:
        """Move the scene with the mouse by (dx, dy) pixels."""
        _, right, up = self.basis()
        per_px = 2.0 * self.half_height(grid) / height
        self.target = self.target - right * dx * per_px + up * dy * per_px

    def zoom_at(self, grid: Grid, factor: float, x: float, y: float, width: float, height: float) -> None:
        """Change the zoom. The point below the pixel (x, y) stays where it is."""
        factor = min(max(self.zoom * factor, 0.2), 200.0) / self.zoom
        _, right, up = self.basis()
        nx = (2.0 * x / width - 1.0) * (width / height)
        ny = 1.0 - 2.0 * y / height
        half = self.half_height(grid)
        self.target = self.target + (nx * right + ny * up) * (half - half / factor)
        self.zoom *= factor

    def look(self, grid: Grid, view: str) -> None:
        """Set a standard direction: 'home', or face on to the plane 'hk', 'hl' or 'kl'.

        A plane view has its first index to the right and its second index up.
        """
        self.target = np.zeros(3)
        self.zoom = 1.0
        if view == 'home':
            self.yaw, self.pitch = math.radians(-55.0), math.radians(22.0)
            return
        first, second = {'hk': (0, 1), 'hl': (0, 2), 'kl': (1, 2)}[view]
        out = np.cross(grid.b[:, first], grid.b[:, second])
        out /= np.linalg.norm(out)
        self.pitch = min(max(math.asin(min(max(out[2], -1.0), 1.0)), -self.LIMIT), self.LIMIT)
        flat = math.hypot(out[0], out[1])
        self.yaw = math.atan2(out[1], out[0]) if flat > 1.0e-9 else -math.pi / 2


# ──────────────────────────────────────────────────────────────────
# The ray march on the CPU (read-out below the mouse, reference for the shader)
# ──────────────────────────────────────────────────────────────────

def sample_limits(shape: tuple[int, ...], lo: npt.ArrayLike, hi: npt.ArrayLike) -> tuple[Array, Array]:
    """Texture coordinates of the centres of the first and the last voxel in the box lo..hi.

    A sample is taken between these limits, so that a point on a face of the
    box has the value of the voxel inside the box, not the mean with the voxel
    outside.
    """
    half = 0.5 / np.array(shape[::-1], dtype=np.float64)
    a, b = np.asarray(lo, float) + half, np.asarray(hi, float) - half
    mid = 0.5 * (a + b)
    return np.minimum(a, mid), np.maximum(b, mid)


def trilinear(data: npt.NDArray[Any], tex: npt.ArrayLike, lo: npt.ArrayLike | None = None,
              hi: npt.ArrayLike | None = None) -> Array:
    """Values at texture coordinates as the GPU gives them.

    Linear filter, coordinates clamped to the edge, and NaN, infinity and
    negative infinity read as 0, 3e38 and 0 (the texture has these values).
    With a box `lo`..`hi`, the coordinates are first clamped to `sample_limits`.
    """
    n = np.array(data.shape, dtype=np.float64)
    tex = np.atleast_2d(np.asarray(tex, dtype=np.float64))
    if lo is not None and hi is not None:
        tex = np.clip(tex, *sample_limits(data.shape, lo, hi))
    idx = tex[:, ::-1] * n - 0.5
    i0 = np.floor(idx).astype(np.int64)
    f = idx - i0
    top = np.array(data.shape) - 1
    out = np.zeros(len(idx))
    for corner in product((0, 1), repeat=3):
        c = np.array(corner)
        ii = np.clip(i0 + c, 0, top)
        w = np.prod(np.where(c == 1, f, 1.0 - f), axis=1)
        v = np.asarray(data[ii[:, 0], ii[:, 1], ii[:, 2]], dtype=np.float64)
        out += w * np.nan_to_num(v, nan=0.0, posinf=3e38, neginf=0.0)
    return out


def _segment(grid: Grid, origin: Array, direction: Array, lo: npt.ArrayLike, hi: npt.ArrayLike,
             ortho: bool) -> tuple[Array, Array, float, float] | None:
    """The part of a ray in the box: (origin, direction) in texture coordinates, t of entry and exit."""
    o = _apply(grid.w2t, origin)[0]
    d = grid.w2t[:3, :3] @ np.asarray(direction, dtype=np.float64)
    d = np.where(np.abs(d) < 1.0e-9, 1.0e-9, d)
    ta, tb = (np.asarray(lo, float) - o) / d, (np.asarray(hi, float) - o) / d
    tn, tf = float(np.minimum(ta, tb).max()), float(np.maximum(ta, tb).min())
    if not ortho:
        tn = max(tn, 0.0)
    return (o, d, tn, tf) if tf > tn else None


def first_hit(data: npt.NDArray[Any], grid: Grid, origin: Array, direction: Array, level: float,
              lo: npt.ArrayLike = (0.0, 0.0, 0.0), hi: npt.ArrayLike = (1.0, 1.0, 1.0),
              step: float = 1.0, floor: float = 1.0e-3, ortho: bool = True) -> dict[str, Any] | None:
    """First point of a ray at or above `level`, found as the shader finds it.

    `data` is the array of the texture (shape `grid.shape`). Returns None, or a
    dict with `t` (distance along the ray in world units), `tex`, `hkl`, `value`
    and `cap` (True if the ray is above the level where it enters the box).
    """
    seg = _segment(grid, origin, direction, lo, hi, ortho)
    if seg is None:
        return None
    o, d, tn, tf = seg
    dt = step / float(np.linalg.norm(d * np.array(grid.shape[::-1], dtype=np.float64)))
    n = int(max((tf - tn) / dt, 0.0)) + 1
    later = tn + dt * np.arange(1, n + 1)
    ts = np.concatenate([[tn + 1.0e-3 * dt], later[later <= tf]])       # the first sample is always taken
    vals = trilinear(data, o + ts[:, None] * d, lo, hi)
    above = np.nonzero(vals >= level)[0]
    if above.size == 0:
        return None
    i = int(above[0])
    if i == 0:
        t, cap = tn, True
    else:
        t_a, t_b = (tn if i == 1 else float(ts[i - 1])), float(ts[i])
        v_a, v_b = float(vals[i - 1]), float(vals[i])
        for _ in range(4):                               # halve the interval that has the level
            t_m = 0.5 * (t_a + t_b)
            v_m = float(trilinear(data, o + t_m * d, lo, hi)[0])
            if v_m >= level:
                t_b, v_b = t_m, v_m
            else:
                t_a, v_a = t_m, v_m
        a, b = math.log(max(v_a, floor)), math.log(max(v_b, floor))
        frac = min(max((math.log(level) - a) / (b - a), 0.0), 1.0) if b != a else 1.0
        t, cap = t_a + frac * (t_b - t_a), False
    tex = o + t * d
    return {'t': t, 'tex': tex, 'hkl': grid.hkl(tex)[0],
            'value': float(trilinear(data, tex, lo, hi)[0]), 'cap': cap}


def ray_max(data: npt.NDArray[Any], grid: Grid, origin: Array, direction: Array,
            lo: npt.ArrayLike = (0.0, 0.0, 0.0), hi: npt.ArrayLike = (1.0, 1.0, 1.0),
            step: float = 1.0, ortho: bool = True) -> dict[str, Any] | None:
    """Largest value along a ray in the box (the projection), with its position.

    As in the shader, the samples are at the entry point and then each `step`
    voxels, and a value below zero counts as zero.
    """
    seg = _segment(grid, origin, direction, lo, hi, ortho)
    if seg is None:
        return None
    o, d, tn, tf = seg
    dt = step / float(np.linalg.norm(d * np.array(grid.shape[::-1], dtype=np.float64)))
    ts = tn + dt * np.arange(int(math.floor((tf - tn) / dt)) + 1)
    vals = np.maximum(trilinear(data, o + ts[:, None] * d, lo, hi), 0.0)
    i = int(np.argmax(vals))
    tex = o + float(ts[i]) * d
    return {'t': float(ts[i]), 'tex': tex, 'hkl': grid.hkl(tex)[0], 'value': float(vals[i])}


# ──────────────────────────────────────────────────────────────────
# Labels of the box
# ──────────────────────────────────────────────────────────────────

def nice_step(units_per_px: float, min_px: float) -> float:
    """Smallest step of 1, 2 or 5 times a power of ten that is `min_px` pixels or more."""
    need = units_per_px * min_px
    if not math.isfinite(need) or need <= 0.0:
        return 1.0
    base = 10.0 ** math.floor(math.log10(need))
    for m in (1.0, 2.0, 5.0, 10.0):
        if m * base >= need * (1.0 - 1.0e-9):
            return m * base
    return 10.0 * base


def _number(value: float, step: float) -> str:
    decimals = max(0, -int(math.floor(math.log10(step) + 1.0e-9)))
    text = f'{value:.{decimals}f}'
    return '0' if float(text) == 0.0 else text


def axis_layout(grid: Grid, cam: Camera, lo: npt.ArrayLike, hi: npt.ArrayLike,
                width: float, height: float) -> list[dict[str, Any]]:
    """Where to write the numbers and the names of h, k and l around the box.

    For each index one edge of the box is used: an edge of the outline, the
    lowest one for an edge that lies flat on the screen and the one most to the
    left for an edge that stands up. An index whose edges are shorter than 40
    pixels gets no numbers. Each entry has `axis`, `ticks` (x, y, text),
    `name` (x, y), `edge` (the two ends in pixels), `step` (of the numbers)
    and `px_per_unit`.
    """
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    centre = cam.project(grid, grid.world(0.5 * (lo + hi)), width, height)[0]
    fwd = cam.basis()[0]
    eye = cam.eye(grid)
    out: list[dict[str, Any]] = []
    for ax in range(3):
        ta = _TEX_OF_AXIS[ax]
        tb, tc = [i for i in range(3) if i != ta]
        best: tuple[Any, ...] | None = None
        for vb, vc in product((0, 1), repeat=2):
            p0, p1 = lo.copy(), hi.copy()
            p0[tb] = p1[tb] = (hi if vb else lo)[tb]
            p0[tc] = p1[tc] = (hi if vc else lo)[tc]
            ends_w = grid.world([p0, p1])
            ends = cam.project(grid, ends_w, width, height)
            mid = ends.mean(axis=0)
            view = fwd if cam.ortho else ends_w.mean(axis=0) - eye
            front = [float(((1 if side else -1) * grid.w2t[t, :3]) @ view) < 0.0
                     for t, side in ((tb, vb), (tc, vc))]
            flat = abs(ends[1, 0] - ends[0, 0]) >= abs(ends[1, 1] - ends[0, 1])
            key = (front[0] != front[1], mid[1] if flat else -mid[0])
            if best is None or key > best[0]:
                best = (key, p0, p1, ends, mid)
        assert best is not None
        _, p0, p1, ends, mid = best
        length = float(np.linalg.norm(ends[1] - ends[0]))
        m0, m1 = float(grid.hkl(p0)[0, ax]), float(grid.hkl(p1)[0, ax])
        entry: dict[str, Any] = {'axis': AXIS_NAMES[ax], 'ticks': [], 'name': None, 'edge': ends,
                                 'step': 1.0, 'px_per_unit': length / max(abs(m1 - m0), 1.0e-12)}
        if length >= 40.0 and abs(m1 - m0) > 1.0e-9:
            # the numbers are beside their edge, on the side away from the box: at a
            # corner of the box the numbers of two axes are then not side by side
            edge = (ends[1] - ends[0]) / length
            away = np.array([edge[1], -edge[0]])
            if away @ (mid - centre) < 0.0:
                away = -away
            step = nice_step(abs(m1 - m0) / length, 46.0)
            first = math.ceil(min(m0, m1) / step - 1.0e-6)
            last = math.floor(max(m0, m1) / step + 1.0e-6)
            marks = np.arange(first, last + 1) * step
            # each number is at the projection of its own point (in perspective the
            # numbers are not at equal distances along the edge)
            on_edge = p0 + ((marks - m0) / (m1 - m0))[:, None] * (p1 - p0)
            places = cam.project(grid, grid.world(on_edge), width, height) + 14.0 * away
            for value, pos in zip(marks, places):
                entry['ticks'].append((float(pos[0]), float(pos[1]), _number(float(value), step)))
            entry['name'] = tuple(float(v) for v in mid + 36.0 * away)
            entry['step'] = step
        out.append(entry)
    # Two axes meet at a corner of the box. If their numbers are less than 22 pixels
    # apart there, the number of the second axis is not written ("6" and "-6" side by side).
    for j in range(1, 3):
        earlier = [t for entry in out[:j] for t in entry['ticks']]
        out[j]['ticks'] = [t for t in out[j]['ticks']
                           if all(math.hypot(t[0] - e[0], t[1] - e[1]) >= 22.0 for e in earlier)]
    return out


def grid_steps(layout: list[dict[str, Any]]) -> tuple[float, float, float]:
    """Distance of the grid lines for h, k and l.

    The lines are at whole Miller indices and 20 pixels apart or more. Each
    number of an axis is on a line: the distance of the lines is the distance
    of the numbers, or a fifth or a half of it.
    """
    steps = []
    for entry in layout:
        label, px = float(entry['step']), float(entry['px_per_unit'])
        best = max(1.0, label)
        for part in (5.0, 2.0):
            step = label / part
            if step >= 1.0 and abs(step - round(step)) < 1.0e-9 and step * px >= 20.0:
                best = step
                break
        steps.append(best)
    return tuple(steps)        # type: ignore[return-value]


def triad(grid: Grid, cam: Camera) -> list[tuple[str, float, float, float]]:
    """Screen directions of a*, b* and c*: (name, x, y, depth). The longest arrow has length 1."""
    _, right, up = cam.basis()
    fwd = cam.basis()[0]
    vecs = [grid.b[:, i] / np.linalg.norm(grid.b[:, i]) for i in range(3)]
    return [(name, float(v @ right), float(-(v @ up)), float(v @ fwd))
            for name, v in zip(('a*', 'b*', 'c*'), vecs)]
