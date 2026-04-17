"""Regression: cleaned rspace3d vs original rspace3d on real data.

Tests:
  1) VolumeData load from example_monoclinic.h5
  2) HK/HL/KL slice extraction (native + regridded)
  3) RSPLayer read from an .img file
"""
from __future__ import annotations

import sys
import importlib
import numpy as np
from pathlib import Path


def load_pkg(path: str, alias: str):
    sys.path.insert(0, path)
    for name in list(sys.modules):
        if name == "rspace3d" or name.startswith("rspace3d."):
            del sys.modules[name]
    mod = importlib.import_module("rspace3d")
    sys.path.pop(0)
    return mod


def test_volume(pkg, label: str, h5_path: Path) -> dict:
    vb = pkg.volume_builder
    vol = vb.load_volume_h5(str(h5_path))
    md = vol.metadata or {}
    out: dict = {
        "shape": vol.intensity.shape,
        "dtype": str(vol.intensity.dtype),
        "sum": float(np.sum(vol.intensity, dtype=np.float64)),
        "nnz": int(np.count_nonzero(vol.intensity)),
        "H0": float(vol.H[0]),
        "K0": float(vol.K[0]),
        "L0": float(vol.L[0]),
        "M_inv": np.asarray(md.get("M_inv")).tolist() if md.get("M_inv") is not None else None,
        "cell_a": md.get("cell_a"),
        "cell_b": md.get("cell_b"),
        "cell_c": md.get("cell_c"),
    }
    # 0=HK (native), 1=HL, 2=KL
    hk, *_ = vb.extract_volume_slice(vol, 0, 0.0)
    out["HK_sum"] = float(np.sum(hk, dtype=np.float64))
    out["HK_shape"] = hk.shape
    hl, *_ = vb.extract_volume_slice(vol, 1, 0.0)
    out["HL_sum"] = float(np.sum(hl, dtype=np.float64))
    out["HL_shape"] = hl.shape
    kl, *_ = vb.extract_volume_slice(vol, 2, 0.0)
    out["KL_sum"] = float(np.sum(kl, dtype=np.float64))
    out["KL_shape"] = kl.shape
    print(f"[{label}] vol shape={out['shape']} sum={out['sum']:.3f}")
    print(f"[{label}]   HK sum={out['HK_sum']:.3f} shape={out['HK_shape']}")
    print(f"[{label}]   HL sum={out['HL_sum']:.3f} shape={out['HL_shape']}")
    print(f"[{label}]   KL sum={out['KL_sum']:.3f} shape={out['KL_shape']}")
    return out


def test_img(pkg, label: str, img_path: Path) -> dict:
    layer = pkg.rsp_reader.read_rsp_layer(str(img_path))
    out: dict = {
        "plane": layer.plane_type,
        "shape": layer.intensity.shape,
        "sum": float(np.sum(layer.intensity, dtype=np.float64)),
        "fixed_value": float(layer.fixed_value),
        "s": float(layer.s),
        "cx": float(layer.cx),
        "cy": float(layer.cy),
        "M_inv": np.asarray(layer.M_inv).tolist(),
    }
    print(f"[{label}] img plane={out['plane']} shape={out['shape']} sum={out['sum']:.3f}")
    return out


def compare(a: dict, b: dict, name: str) -> bool:
    ok = True
    for k in a:
        va, vb_ = a[k], b[k]
        if isinstance(va, (list, tuple)) and va and isinstance(va[0], (list, tuple)):
            va_arr = np.asarray(va, dtype=float)
            vb_arr = np.asarray(vb_, dtype=float)
            if not np.allclose(va_arr, vb_arr, atol=1e-10):
                print(f"  MISMATCH {name}.{k}: max|d|={np.max(np.abs(va_arr-vb_arr))}")
                ok = False
        elif isinstance(va, float):
            if not np.isclose(va, vb_, atol=1e-6, rtol=1e-9):
                print(f"  MISMATCH {name}.{k}: {va} vs {vb_}")
                ok = False
        elif va != vb_:
            print(f"  MISMATCH {name}.{k}: {va!r} vs {vb_!r}")
            ok = False
    return ok


if __name__ == "__main__":
    H5 = Path(r"C:/Users/Milos/Desktop/rspace3d/example_data/example_monoclinic.h5")
    IMG = Path(r"F:/CrysAlis_3D_reconstructions/I15_Nov_24/FAPbBr3_T_Dep_Cbf/FAPbBr3_230K_HK1.5.img")

    print("Loading ORIGINAL rspace3d...")
    orig = load_pkg(r"C:/Users/Milos/Desktop/rspace3d", "rspace3d_orig")
    o_vol = test_volume(orig, "orig", H5)
    o_img = test_img(orig, "orig", IMG) if IMG.exists() else None

    print()
    print("Loading CLEANED rspace3d...")
    clean = load_pkg(r"C:/Users/Milos/Desktop/rspace3d_cleanup", "rspace3d_clean")
    c_vol = test_volume(clean, "clean", H5)
    c_img = test_img(clean, "clean", IMG) if IMG.exists() else None

    print()
    print("=== Comparison ===")
    all_ok = True
    all_ok &= compare(o_vol, c_vol, "vol")
    if o_img is not None and c_img is not None:
        all_ok &= compare(o_img, c_img, "img")

    if all_ok:
        print("PASS: cleaned output matches original on all tested fields.")
        sys.exit(0)
    else:
        print("FAIL: mismatches above.")
        sys.exit(1)
