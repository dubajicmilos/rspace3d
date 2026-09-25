"""Instrument-agnostic geometry calibration from the CrysAlisPro peak-hunt tabbin.

`geometry_from_tabbin(folder)` fits the FULL experiment geometry - detector pose
(beam centre, distance, orientation/handedness), oscillation axis + sense, and the
crystal orientation R0 - directly from CrysAlisPro's indexed peaks
(pixel <-> frame <-> reciprocal vector). Nothing instrument-specific is assumed, so
it works across detectors/beamlines: an Eiger at 85 mm and a Pilatus at ~649 mm are
handled by the same code. Assumes a flat detector, single-axis rotation scan, and a
monochromatic beam (the standard rotation-experiment case).

The header geometry is used only as a starting point and may be wrong (I15's header
distance was off by 1.75x) - the tabbin correspondences are the reference.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation as _Rot

from .geometry import (FlatDetector, find_crysalis_par, read_crysalis_par,
                       find_cbf_frames, _kabsch)
from .tabbin import read_tabbin, find_tabbin

_Z = np.array([0.0, 0.0, 1.0])
_X = np.array([1.0, 0.0, 0.0])
_Y = np.array([0.0, 1.0, 0.0])
# the 8 in-plane detector orientations (col-axis dir, row-axis dir) for beam=+z
_ORIENTS = [(_X, _Y), (_X, -_Y), (-_X, _Y), (-_X, -_Y),
            (_Y, _X), (_Y, -_X), (-_Y, _X), (-_Y, -_X)]
# oscillation-axis initial guesses (unit vectors): +Y, +X, +Z
_OSC_INITS = [np.array([0., 1, 0]), np.array([1., 0, 0]), np.array([0., 0, 1])]


@dataclass
class FittedGeometry:
    detector: FlatDetector
    UB: np.ndarray
    R0: np.ndarray
    osc_axis: tuple
    sense: int
    phi0: float
    increment: float
    rms: float                 # 1/A, fit residual over the fitted tabbin peaks
    n_peaks: int               # peaks used by the fit
    median_hkl_dev: float      # rlu, fitted peaks -> integer hkl through the detector
    median_hkl_dev_heldout: float = float("nan")   # rlu, peaks the fit never saw
    n_heldout: int = 0
    mapping_margin: float = float("inf")           # best / runner-up pixel-mapping score
    pixel_mapping: str = ""
    par_path: str = ""
    tabbin_path: str = ""


# Evidence gates. MAPPING_MIN_MARGIN: the correct tabbin->CBF pixel mapping
# scores ~40x above the runner-up on real spot patterns (4e4x on I19-2); a
# spot-free or mirror-symmetric pattern gives ~1x, where the pick is arbitrary
# and a wrong branch smears the reconstruction by ~0.5 rlu. MIN_ROTATION_SPAN_DEG:
# the oscillation axis only enters the residual through the rotation between
# frames, so peaks from a single frame leave it at its initial guess.
MAPPING_MIN_MARGIN = 3.0
MIN_ROTATION_SPAN_DEG = 5.0
HKL_DEV_LIMIT = 0.1            # rlu, median deviation from integer hkl


def _rotate(v, axis, ang):     # Rodrigues, per-row angle
    c = np.cos(ang)[:, None]; s = np.sin(ang)[:, None]
    return (v * c + np.cross(np.broadcast_to(axis, v.shape), v) * s
            + axis[None, :] * (v @ axis)[:, None] * (1 - c))


def _sph(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])


def _ground_mapping(pk, keep, frames, nx, ny, n_strong=60):
    """Resolve how the tabbin (px,py) map to CBF-frame (col in 1..nx, row in 1..ny)
    by testing all 8 transpose/flip options against the actual detector intensities:
    the correct mapping puts the peaks on real bright pixels. Returns (col, row, name)
    with col,row 1-based arrays for the kept peaks."""
    import fabio

    idx = np.where(keep)[0]
    strong = idx[np.argsort(pk["intensity"][idx])[::-1][:n_strong]]
    spx, spy, sfr = pk["px"][strong], pk["py"][strong], pk["frame"][strong]

    def make(tr, fx, fy):
        def f(px, py):
            a, b = (py, px) if tr else (px, py)
            return ((nx + 1 - a) if fx else a), ((ny + 1 - b) if fy else b)
        return f

    maps = [(f"{'T' if tr else '-'}{'X' if fx else '-'}{'Y' if fy else '-'}", make(tr, fx, fy))
            for tr in (0, 1) for fx in (0, 1) for fy in (0, 1)]
    scores = np.zeros(len(maps))
    for fnum in np.unique(sfr):
        img = fabio.open(frames[int(fnum) - 1]).data
        sel = sfr == fnum
        for mi, (_, f) in enumerate(maps):
            c, r = f(spx[sel].astype(int), spy[sel].astype(int))
            for cc, rr in zip(c, r):
                ci, ri = int(cc) - 1, int(rr) - 1
                if 0 <= ri < ny and 0 <= ci < nx:
                    w = img[max(0, ri-3):ri+4, max(0, ci-3):ci+4]
                    if w.size:
                        scores[mi] += float(w.max())
    order = np.argsort(scores)[::-1]
    best, runner_up = scores[order[0]], scores[order[1]]
    margin = float("inf") if runner_up <= 0 else float(best / runner_up)
    if best <= 0 or margin < MAPPING_MIN_MARGIN:
        raise RuntimeError(
            f"pixel-mapping evidence is not discriminating: {maps[order[0]][0]} scored "
            f"{best:.0f} against runner-up {maps[order[1]][0]} {runner_up:.0f} "
            f"(margin {margin:.2f}x < {MAPPING_MIN_MARGIN}x). The strongest tabbin peaks "
            "do not land on bright CBF pixels for one mapping only; check the frames/tabbin.")
    name, f = maps[int(order[0])]
    col, row = f(pk["px"][keep].astype(float), pk["py"][keep].astype(float))
    return col, row, name, margin


def geometry_from_tabbin(folder, *, par_path=None, tab_path=None,
                         clean_tol=0.02, rms_ok=0.01, log=None):
    """Fit (detector, UB, R0, osc_axis, sense, phi0, increment) from the tabbin.

    par_path / tab_path : explicit CrysAlisPro par and peak-hunt tabbin; by
        default the par is discovered in `folder` and the tabbin is the one that
        belongs to that par (`find_tabbin(folder, par_path=...)`).

    The fit uses ~80% of the cleanly indexed peaks; the remaining ~20%, spread
    over the scan, validate it (median hkl deviation through the fitted
    geometry, limit `HKL_DEV_LIMIT`). Returns a FittedGeometry. Raises if the
    evidence is weak or no good fit is found (honest failure).
    """
    say = log or (lambda *a: None)
    par = par_path or find_crysalis_par(folder)
    tab = tab_path or (find_tabbin(folder, par_path=par) if par else None)
    if par is None or tab is None:
        raise FileNotFoundError(f"need a *_cracker.par and a *_peakhunt.tabbin in {folder}")
    frames, nums = find_cbf_frames(folder)
    det0, ang0, _ = FlatDetector.from_eiger_cbf(frames[0])
    ny, nx = det0.shape; ps, wl = det0.pixel_size, det0.wavelength
    phi0, incr = ang0["Phi"], ang0["Angle_increment"]
    UB, _ = read_crysalis_par(par)

    pk = read_tabbin(tab)
    hklf = (np.linalg.inv(UB * wl) @ pk["d"].T).T          # hkl (dxdydz = wl*UB*hkl)
    keep = np.abs(hklf - np.round(hklf)).max(1) < clean_tol
    if keep.sum() < 20:
        raise RuntimeError(f"only {keep.sum()} cleanly-indexed tabbin peaks; cannot calibrate")
    # Ground the tabbin px/py -> CBF-frame (col in 1..nx, row in 1..ny) mapping against
    # the actual detector intensities: the peaks must land on real bright pixels. This
    # resolves the transpose/flip ambiguity that peak *positions* alone cannot (a
    # near-square detector or a flipped readout otherwise picks a self-consistent but
    # wrong branch).
    col_all, row_all, mp_name, margin = _ground_mapping(pk, keep, frames, nx, ny)
    say(f"pixel mapping grounded to CBF intensity: {mp_name} (margin {margin:.3g}x)")
    frame_all = pk["frame"][keep]
    g_all = (UB @ np.round(hklf[keep]).T).T                # crystal-frame reciprocal (1/d)
    dphi_all = (frame_all - 1) * incr                      # deg since frame 1
    span = float(np.ptp(dphi_all))
    if span < MIN_ROTATION_SPAN_DEG:
        raise RuntimeError(
            f"the indexed tabbin peaks span only {span:.2f} deg of rotation; the "
            f"oscillation axis cannot be constrained (need >= {MIN_ROTATION_SPAN_DEG} deg). "
            "Run the peak hunt over the whole scan.")
    # Held-out split: every 5th peak in scan order is kept out of the fit.
    order = np.argsort(frame_all, kind="stable")
    held = np.zeros(len(order), bool)
    held[order[4::5]] = True
    fit = ~held
    col, row, frame, g, dphi = (a[fit] for a in (col_all, row_all, frame_all, g_all, dphi_all))
    say(f"tabbin: {pk['n']} peaks, {keep.sum()} cleanly indexed over "
        f"{len(set(frame_all))} frames; fit on {fit.sum()}, held out {held.sum()}")

    fref = np.bincount(frame).argmax()

    def resid(p, Fc, Fr, sense):
        bcc, bcr, dist, rx, ry, rz, oth, oph = p
        R0 = _Rot.from_rotvec([rx, ry, rz]).as_matrix()
        osc = _sph(oth, oph)
        u = (col - bcc) * ps; v = (row - bcr) * ps
        P = u[:, None]*Fc + v[:, None]*Fr + dist*_Z
        robs = P/np.linalg.norm(P, axis=1)[:, None]/wl - _Z/wl
        rpred = _rotate((R0 @ g.T).T, osc, np.radians(sense*dphi))
        return (robs - rpred).ravel()

    best = None
    osc_list = _OSC_INITS[:1]                              # start with +Y
    tried_all = False
    while True:
        for Fc, Fr in _ORIENTS:
            for sense in (1, -1):
                # R0 init: Kabsch on the reference frame with header geometry
                u = (col-det0.beam_center[0])*ps; v = (row-det0.beam_center[1])*ps
                P = u[:, None]*Fc + v[:, None]*Fr + det0.distance*_Z
                rref = P/np.linalg.norm(P, axis=1)[:, None]/wl - _Z/wl
                m = frame == fref
                R0i = _kabsch(g[m].T, rref[m].T) if m.sum() >= 3 else np.eye(3)
                r0v = _Rot.from_matrix(R0i).as_rotvec()
                for osc0 in osc_list:
                    th = np.arccos(np.clip(osc0[1], -1, 1)); ph = np.arctan2(osc0[2], osc0[0])
                    p0 = [det0.beam_center[0], det0.beam_center[1], det0.distance,
                          *r0v, th, ph]
                    try:
                        sol = least_squares(resid, p0, args=(Fc, Fr, sense),
                                            loss="soft_l1", f_scale=0.02, max_nfev=250)
                    except Exception:
                        continue
                    rr = float(np.sqrt(np.mean(sol.fun**2)))
                    if best is None or rr < best[0]:
                        best = (rr, sol.x, (Fc, Fr, sense))
        if best and best[0] <= rms_ok or tried_all:
            break
        osc_list = _OSC_INITS                              # retry with X,Z osc inits
        tried_all = True
    say(f"best fit rms {best[0]:.4f} 1/A")

    rr, p, (Fc, Fr, sense) = best
    bcc, bcr, dist = p[:3]
    R0 = _Rot.from_rotvec(p[3:6]).as_matrix()
    osc = _sph(p[6], p[7]); osc = osc/np.linalg.norm(osc)
    det = FlatDetector(distance=dist, pixel_size=ps, beam_center=(bcc-1, bcr-1),
                       shape=(ny, nx), wavelength=wl, beam=_Z, fast=Fc, slow=Fr)

    # Validate: do the tabbin peaks map to integer hkl through this detector?
    # Both the fitted subset and the peaks the fit never saw must pass.
    from .geometry import _axis_rot

    def hkl_dev(cols, rows, dphis, stride):
        devs = []
        for i in range(0, len(cols), stride):
            rlab = det.scattering_vectors([[cols[i]-1, rows[i]-1]])[0]
            Rn = _axis_rot(tuple(osc), sense*dphis[i]) @ R0
            h = np.linalg.inv(Rn @ UB) @ rlab
            devs.append(np.linalg.norm(h - np.round(h)))
        return float(np.median(devs)) if devs else float("nan")

    med = hkl_dev(col, row, dphi, max(1, len(col)//400))
    med_held = hkl_dev(col_all[held], row_all[held], dphi_all[held], 1)
    say(f"validation: fitted peaks -> integer hkl, median dev {med:.4f} rlu; "
        f"held-out {int(held.sum())} peaks -> median dev {med_held:.4f} rlu")
    if med > HKL_DEV_LIMIT:
        raise RuntimeError(f"tabbin geometry fit failed self-check (median hkl dev {med:.3f} rlu)")
    if held.any() and med_held > HKL_DEV_LIMIT:
        raise RuntimeError(
            f"tabbin geometry does not generalise: held-out peaks deviate "
            f"{med_held:.3f} rlu from integer hkl (fitted peaks {med:.3f} rlu)")

    return FittedGeometry(det, np.asarray(UB), R0, tuple(float(x) for x in osc),
                          int(sense), float(phi0), float(incr), rr, int(fit.sum()), med,
                          median_hkl_dev_heldout=med_held, n_heldout=int(held.sum()),
                          mapping_margin=margin, pixel_mapping=mp_name,
                          par_path=str(par), tabbin_path=str(tab))
