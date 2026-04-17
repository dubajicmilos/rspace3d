"""Process exp_172 unwarp folder with cleaned rspace3d.

No binning, Laue m-3m, 3 sigma outlier rejection.
Saves to new filenames (does not overwrite existing h5 files).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, r"C:/Users/Milos/Desktop/rspace3d_cleanup")

import rspace3d  # noqa: F401  triggers fabio detector-type-8 patch
from rspace3d.volume_builder import (
    load_unwarp_folder,
    reject_outliers,
    symmetrize_volume,
    save_volume_h5,
)


FOLDER = Path(r"C:/Users/Milos/Downloads/exp_172/exp_172/unwarp")
RAW_OUT = FOLDER / "exp_172_unwarp_raw_cleaned.h5"
SYM_OUT = FOLDER / "exp_172_unwarp_sym_mbar3m_cleaned.h5"

assert not RAW_OUT.exists(), f"would overwrite {RAW_OUT}"
assert not SYM_OUT.exists(), f"would overwrite {SYM_OUT}"


_last = [0]
def progress(done: int, total: int) -> None:
    # throttle prints to every 10% to keep stdout readable
    pct = (done * 10) // max(total, 1)
    if pct != _last[0]:
        _last[0] = pct
        print(f"    {done}/{total}  ({pct * 10}%)")


t0 = time.time()
print("[1/4] load_unwarp_folder (bin_xy=1)...")
vol = load_unwarp_folder(str(FOLDER), bin_xy=1, progress_callback=progress)
print(f"    shape={vol.intensity.shape} dtype={vol.intensity.dtype}  "
      f"sum={vol.intensity.sum():,}  elapsed={time.time()-t0:.1f}s")

t1 = time.time()
print(f"[2/4] save raw -> {RAW_OUT.name}")
save_volume_h5(str(RAW_OUT), vol, compression="gzip")
print(f"    elapsed={time.time()-t1:.1f}s")

_last[0] = 0
t2 = time.time()
print("[3/4] reject_outliers(sigma=3, n_iter=2, laue=m-3m)")
vol = reject_outliers(vol, laue_group="m-3m", sigma=3.0,
                      n_iter=2, progress_callback=progress)
print(f"    elapsed={time.time()-t2:.1f}s  sum_after={vol.intensity.sum():,}")

_last[0] = 0
t3 = time.time()
print("[4/4] symmetrize_volume(laue=m-3m) + save")
vol = symmetrize_volume(vol, laue_group="m-3m", progress_callback=progress)
save_volume_h5(str(SYM_OUT), vol, compression="gzip")
print(f"    elapsed={time.time()-t3:.1f}s  sum_final={vol.intensity.sum():,}")

print(f"\nDONE. Total={time.time()-t0:.1f}s")
print(f"  raw:  {RAW_OUT}")
print(f"  sym:  {SYM_OUT}")
