"""Validate cleaned rspace3d on exp_172 data.

Loads raw + symmetrised h5, extracts HK0 and HK0.2 slices with each
package version, compares numerically, and saves side-by-side plots.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


H5_RAW = r"C:/Users/Milos/Downloads/exp_172/exp_172/unwarp/exp_172_unwarp_raw.h5"
H5_SYM = r"C:/Users/Milos/Downloads/exp_172/exp_172/unwarp/exp_172_unwarp_sym_mbar3m.h5"
OUT_DIR = Path(r"C:/Users/Milos/Desktop/rspace3d_cleanup/docs/exp_172_validation")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_pkg(path: str):
    """Fresh import of rspace3d from a specific source tree."""
    sys.path.insert(0, path)
    for name in list(sys.modules):
        if name == "rspace3d" or name.startswith("rspace3d."):
            del sys.modules[name]
    mod = importlib.import_module("rspace3d")
    sys.path.pop(0)
    return mod


def extract_slice(pkg, vol, plane_index: int, target: float):
    # 0=HK(fix L), 1=HL(fix K), 2=KL(fix H)
    s, x_ax, y_ax, x_lbl, y_lbl, fix_lbl, actual, n_slices = (
        pkg.volume_builder.extract_volume_slice(vol, plane_index, target, int_range=0.0)
    )
    return {
        "data": np.asarray(s),
        "x_ax": np.asarray(x_ax),
        "y_ax": np.asarray(y_ax),
        "x_lbl": x_lbl,
        "y_lbl": y_lbl,
        "fix_lbl": fix_lbl,
        "actual": actual,
        "n_slices": n_slices,
    }


def plot_and_compare(orig_pkg, clean_pkg, h5_path: str, label: str):
    o_vol = orig_pkg.volume_builder.load_volume_h5(h5_path)
    c_vol = clean_pkg.volume_builder.load_volume_h5(h5_path)

    targets = [("HK0", 0.0), ("HK0.2", 0.2)]
    for tag, l_val in targets:
        o = extract_slice(orig_pkg, o_vol, 0, l_val)
        c = extract_slice(clean_pkg, c_vol, 0, l_val)

        # numerical compare
        if o["data"].shape == c["data"].shape:
            diff = np.asarray(o["data"], dtype=np.float64) - np.asarray(c["data"], dtype=np.float64)
            max_abs = float(np.max(np.abs(diff)))
            rms = float(np.sqrt(np.mean(diff**2)))
        else:
            max_abs, rms = float("nan"), float("nan")

        print(f"[{label} {tag}] shape={o['data'].shape} actual_l={o['actual']:.4f} "
              f"sum_orig={o['data'].sum():.3f} sum_clean={c['data'].sum():.3f} "
              f"max|d|={max_abs:.3e} rms={rms:.3e}")

        # plot cleaned version (log scale, clipped to robust percentile)
        d = c["data"]
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            vmin = max(np.percentile(d[valid], 1), 1e-3)
            vmax = np.percentile(d[valid], 99.5)
        else:
            vmin, vmax = 1e-3, 1.0

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        im = ax.imshow(
            d,
            origin="lower",
            extent=(c["x_ax"][0], c["x_ax"][-1], c["y_ax"][0], c["y_ax"][-1]),
            cmap="viridis",
            norm=matplotlib.colors.LogNorm(vmin=vmin, vmax=vmax),
            aspect="equal",
        )
        ax.set_xlabel(c["x_lbl"])
        ax.set_ylabel(c["y_lbl"])
        ax.set_title(
            f"exp_172 {label} — {tag}\n"
            f"{c['fix_lbl']}={c['actual']:.4f}  "
            f"max|Δorig|={max_abs:.2e}"
        )
        plt.colorbar(im, ax=ax, label="Intensity (log)")
        out = OUT_DIR / f"exp_172_{label}_{tag}.png"
        plt.tight_layout()
        plt.savefig(out, dpi=130)
        plt.close(fig)
        print(f"    saved: {out}")


if __name__ == "__main__":
    print("Loading original package...")
    orig = load_pkg(r"C:/Users/Milos/Desktop/rspace3d")
    print("Loading cleaned package...")
    clean = load_pkg(r"C:/Users/Milos/Desktop/rspace3d_cleanup")

    print("\n=== RAW volume ===")
    plot_and_compare(orig, clean, H5_RAW, "raw")

    print("\n=== SYMMETRISED volume ===")
    plot_and_compare(orig, clean, H5_SYM, "sym_mbar3m")

    print(f"\nPlots saved to: {OUT_DIR}")
