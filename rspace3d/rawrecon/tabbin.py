"""Reader for CrysAlisPro peak-hunt `.tabbin` files (the indexed reflection table).

Binary, little-endian. Layout (version 2-16):
  * header: 312 bytes; uint32 @0 = number of peaks, uint16 @10 = version.
  * then N records of 168 bytes each:
        f64 @0,8,16  dx,dy,dz    reciprocal vector, wavelength-scaled, CRYSTAL frame
                                 (|d| = wl/d-spacing; hkl = (UB*wl)^-1 . d, UB in 1/d)
        f64 @24      dlength     = |d|
        u32 @32      intensity
        u16 @44,46   px, py      detector pixel, 0-based (CrysAlis shows 1-based, +1)
        f32 @48,52   centroid x,y (sub-pixel offset)
        i16 @160     frame       1-based frame number

This is the ground truth used by rawrecon.calibrate.geometry_from_tabbin to fit the
experiment geometry (instrument-agnostic). Reference format tool:
github.com/DESY-Petra-III/CrysalisTabbiner.
"""
from __future__ import annotations

import glob
import os

import numpy as np

_HEADER = 312
_RECSIZE = 168
_DTYPE = np.dtype([
    ("d", "<f8", 3), ("dlen", "<f8"), ("intensity", "<u4"),
    ("_pad1", "V8"), ("px", "<u2"), ("py", "<u2"),
    ("cenx", "<f4"), ("ceny", "<f4"), ("_pad2", "V104"),
    ("frame", "<i2"), ("_pad3", "V2"), ("stamp", "<u4"),
])
assert _DTYPE.itemsize == _RECSIZE, _DTYPE.itemsize


def find_tabbin(folder, par_path=None):
    """Locate the peak-hunt tabbin that belongs to the par in use.

    par_path : the CrysAlisPro par the calibration will use (default: the one
        `find_crysalis_par(folder)` returns). The table is looked for as
        `<base>_peakhunt.tabbin` next to that par first, then in `folder`
        (a par in a subfolder must not pick up an unrelated run's table that
        happens to sit in the dataset folder). Without a base-name match the
        single remaining `*_peakhunt.tabbin` in `folder` is used; several
        unrelated candidates are ambiguous and raise. Intermediate
        `*process_action*` tables are ignored.
    """
    from .geometry import find_crysalis_par

    par = par_path or find_crysalis_par(folder)
    if par:
        base = os.path.basename(par).replace("_cracker.par", "").replace(".par", "")
        for where in (os.path.dirname(par), folder):
            cand = os.path.join(where, base + "_peakhunt.tabbin")
            if os.path.isfile(cand):
                return cand
    hits = [h for h in glob.glob(os.path.join(folder, "*_peakhunt.tabbin"))
            if "process_action" not in os.path.basename(h).lower()]
    if not hits:
        hits = glob.glob(os.path.join(folder, "*_peakhunt.tabbin"))
    if not hits:
        return None
    if len(hits) > 1 and par:
        raise ValueError(
            f"no *_peakhunt.tabbin matches {os.path.basename(par)} and several unrelated "
            f"tables exist in {folder}: {sorted(os.path.basename(h) for h in hits)}; "
            "pass tab_path= explicitly")
    return max(hits, key=os.path.getsize)


def read_tabbin(path):
    """Parse a peak-hunt tabbin. Returns a dict of arrays: d (N,3), dlen, intensity,
    px, py (1-based pixel), cenx, ceny, frame (1-based), plus 'version'.

    Raises ValueError on an unrecognised layout (size mismatch) so a format change
    fails loudly rather than silently mis-reading.
    """
    with open(path, "rb") as f:
        data = f.read()
    n = int.from_bytes(data[0:4], "little")
    version = int.from_bytes(data[10:12], "little")
    if not (2 <= version <= 16):
        raise ValueError(f"{path}: unexpected tabbin version {version}")
    need = _HEADER + n * _RECSIZE
    if len(data) < need or n <= 0:
        raise ValueError(f"{path}: size {len(data)} too small for {n} peaks "
                         f"(need >= {need}); tabbin layout may differ for this version")
    r = np.frombuffer(data, _DTYPE, n, _HEADER)
    return {
        "d": np.ascontiguousarray(r["d"]),
        "dlen": r["dlen"].copy(),
        "intensity": r["intensity"].copy(),
        "px": r["px"].astype(np.int32) + 1,     # -> 1-based (CrysAlis convention)
        "py": r["py"].astype(np.int32) + 1,
        "cenx": r["cenx"].copy(), "ceny": r["ceny"].copy(),
        "frame": r["frame"].astype(np.int32),
        "version": version, "n": n,
    }
