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
    rms: float                 # 1/A, residual of the fit that gave the geometry
    n_peaks: int               # peaks of that fit (the last pass if the second fit was used)
    median_hkl_dev: float      # rlu, fitted clean peaks -> integer hkl through the detector
    median_hkl_dev_heldout: float = float("nan")   # rlu, peaks the fit never saw
    n_heldout: int = 0
    mapping_margin: float = float("inf")           # best / runner-up pixel-mapping score
    pixel_mapping: str = ""
    par_path: str = ""
    tabbin_path: str = ""
    centroids: bool = False        # the fit used the sub-pixel centroids of the tabbin
    refined: bool = False          # the result is that of the second fit
    n_peaks_refined: int = 0       # peaks in the last pass of the second fit
    axis_change_deg: float = 0.0   # angle between the axis of the first and of the second fit


# Evidence gates. MAPPING_MIN_MARGIN: the correct tabbin->CBF pixel mapping
# scores ~40x above the runner-up on real spot patterns (4e4x on I19-2); a
# spot-free or mirror-symmetric pattern gives ~1x, where the pick is arbitrary
# and a wrong branch smears the reconstruction by ~0.5 rlu. MIN_ROTATION_SPAN_DEG:
# the oscillation axis only enters the residual through the rotation between
# frames, so peaks from a single frame leave it at its initial guess.
MAPPING_MIN_MARGIN = 3.0
MIN_ROTATION_SPAN_DEG = 5.0
HKL_DEV_LIMIT = 0.1            # rlu, median deviation from integer hkl

# Second fit. REFINE_SCHEDULE: the limit of each pass in rlu. A peak is in a
# pass if it is nearer than the limit to an integer hkl through the geometry
# of the pass before. REFINE_STRONG_FRACTION: the second fit uses only this
# part of the table, the strongest peaks by rank. The weak entries of a table
# are tails of peaks and diffuse maxima. Their positions are not exact: a fit
# with them moved the strong peaks away from integer hkl on one of five test
# datasets. REFINE_CHECK_TOL: the strong peaks that the second fit does not use
# and that are nearer than this to an integer hkl after the first fit are the
# check peaks. REFINE_CONFIDENCE and
# REFINE_RESAMPLES: the second fit is used only if the intensity-weighted mean
# distance of the check peaks from integer hkl is smaller with it, in at least
# this part of the bootstrap resamples. On 16 datasets this rule used the
# second fit for no dataset with a worse reconstruction. A median distance in
# rlu, without weights, used it for two such datasets.
REFINE_SCHEDULE = (0.25, 0.15, 0.10, 0.07, 0.05)
REFINE_STRONG_FRACTION = 0.30
REFINE_MIN_PEAKS = 30
REFINE_F_SCALE = 0.002         # 1/A, scale of the soft_l1 loss
REFINE_CHECK_TOL = 0.1         # rlu
REFINE_CONFIDENCE = 0.95
REFINE_RESAMPLES = 1000


def _rotate(v, axis, ang):     # Rodrigues, per-row angle
    c = np.cos(ang)[:, None]; s = np.sin(ang)[:, None]
    return (v * c + np.cross(np.broadcast_to(axis, v.shape), v) * s
            + axis[None, :] * (v @ axis)[:, None] * (1 - c))


def _sph(th, ph):
    return np.array([np.sin(th) * np.cos(ph), np.cos(th), np.sin(th) * np.sin(ph)])


# Fit parameters p = (beam-centre col, beam-centre row, distance, R0 rotation
# vector (3), polar and azimuth angle of the oscillation axis).

def _lab_vectors(p, col, row, Fc, Fr, ps, wl):
    """Scattering vector (1/A) of each pixel (col, row; 1-based) for the detector of `p`."""
    u = (col - p[0]) * ps; v = (row - p[1]) * ps
    P = u[:, None]*Fc + v[:, None]*Fr + p[2]*_Z
    return P/np.linalg.norm(P, axis=1)[:, None]/wl - _Z/wl


def _model_vectors(p, g, dphi, sense):
    """Lab scattering vector that `p` gives for the crystal-frame vectors `g`
    at the rotation angles `dphi` (deg)."""
    R0 = _Rot.from_rotvec(p[3:6]).as_matrix()
    return _rotate((R0 @ g.T).T, _sph(p[6], p[7]), np.radians(sense*dphi))


def _hkl_of_peaks(p, col, row, dphi, UB, Fc, Fr, sense, ps, wl):
    """Fractional hkl of each peak (pixel and rotation angle) through the geometry of `p`."""
    R0 = _Rot.from_rotvec(p[3:6]).as_matrix()
    back = _rotate(_lab_vectors(p, col, row, Fc, Fr, ps, wl), _sph(p[6], p[7]),
                   -np.radians(sense*dphi))
    return np.linalg.solve(R0 @ UB, back.T).T


def _refine_reselect(p, col, row, dphi, usable, UB, Fc, Fr, sense, ps, wl,
                     schedule=REFINE_SCHEDULE):
    """Fit `p` again to the peaks of the table (arrays `col`, `row`, `dphi`).

    Each pass gives each peak the nearest integer hkl through the geometry
    of the pass before. The pass then fits the peaks of `usable` that are
    nearer than its limit to that hkl. A peak that the first selection did
    not keep can thus come into the fit. Returns (p, n, rms): the parameters,
    the number of peaks of the last pass and the rms residual (1/A) of the
    last pass. Returns (None, 0, nan) if a pass has fewer than
    REFINE_MIN_PEAKS peaks."""
    p = np.asarray(p, float)
    n_sel, fun = 0, None
    for limit in schedule:
        hkl = _hkl_of_peaks(p, col, row, dphi, UB, Fc, Fr, sense, ps, wl)
        n = np.round(hkl)
        sel = usable & (np.abs(hkl - n).max(1) < limit) & np.any(n != 0, axis=1)
        n_sel = int(sel.sum())
        if n_sel < REFINE_MIN_PEAKS:
            return None, 0, float("nan")
        c, r, d, g = col[sel], row[sel], dphi[sel], (UB @ n[sel].T).T

        def fun(q, c=c, r=r, d=d, g=g):
            return (_lab_vectors(q, c, r, Fc, Fr, ps, wl) - _model_vectors(q, g, d, sense)).ravel()
        p = least_squares(fun, p, loss="soft_l1", f_scale=REFINE_F_SCALE, x_scale="jac",
                          max_nfev=200).x
    return p, n_sel, float(np.sqrt(np.mean(fun(p) ** 2)))


def _strongest(intensity, fraction):
    """Mask of the `fraction` strongest peaks of the table, by rank.

    The mask has ceil(fraction * N) peaks also if many intensities are equal.
    Equal intensities keep the order of the table."""
    intensity = np.asarray(intensity, float)
    mask = np.zeros(len(intensity), bool)
    mask[np.argsort(-intensity, kind="stable")[:int(np.ceil(fraction * len(intensity)))]] = True
    return mask


def _second_fit_is_better(a1, a2, w, *, resamples=REFINE_RESAMPLES,
                          confidence=REFINE_CONFIDENCE):
    """Compare two geometries on the check peaks with a paired bootstrap.

    a1, a2 : distance (1/A) of each check peak from its integer hkl with the
        first and with the second geometry.
    w : intensity of each check peak (the weight).

    The statistic is the intensity-weighted mean distance. The reconstruction
    adds the intensity of each pixel to its voxel, so a strong peak far from
    its node blurs the volume more than a weak peak. Each resample draws the
    peaks with replacement and keeps the pairs. The seed is fixed, so the
    result is the same in each run.

    Returns (better, mean_1, mean_2, fraction): `fraction` is the part of the
    resamples in which the second geometry has the smaller weighted mean.
    `better` is True if the second mean is smaller and `fraction` is at least
    `confidence`."""
    a1, a2, w = (np.asarray(x, float) for x in (a1, a2, w))
    m1, m2 = float(np.sum(w * a1) / np.sum(w)), float(np.sum(w * a2) / np.sum(w))
    rng = np.random.default_rng(0)
    gain = np.empty(resamples)
    for s in range(0, resamples, 100):                 # blocks of 100 resamples (memory)
        idx = rng.integers(0, len(a1), (min(100, resamples - s), len(a1)))
        ww = w[idx]
        gain[s:s + len(idx)] = (np.sum(ww * a1[idx], 1) - np.sum(ww * a2[idx], 1)) / np.sum(ww, 1)
    fraction = float(np.mean(gain > 0))
    return bool(m2 < m1 and fraction >= confidence), m1, m2, fraction


_MAP_NAMES = ("---", "--Y", "-X-", "-XY", "T--", "T-Y", "TX-", "TXY")


def _pixel_map(name, nx, ny):
    """The function (px, py) -> (col, row) of the mapping `name`:
    'T' = transpose, 'X' = flip of the columns, 'Y' = flip of the rows."""
    tr, fx, fy = name[0] == "T", name[1] == "X", name[2] == "Y"

    def f(px, py):
        a, b = (py, px) if tr else (px, py)
        return ((nx + 1 - a) if fx else a), ((ny + 1 - b) if fy else b)
    return f


def _ground_mapping(pk, keep, frames, nx, ny, n_strong=60):
    """Resolve how the tabbin (px,py) map to CBF-frame (col in 1..nx, row in 1..ny)
    by testing all 8 transpose/flip options against the actual detector intensities:
    the correct mapping puts the peaks on real bright pixels. Returns (col, row, name)
    with col,row 1-based arrays for the kept peaks."""
    import fabio

    idx = np.where(keep)[0]
    strong = idx[np.argsort(pk["intensity"][idx])[::-1][:n_strong]]
    spx, spy, sfr = pk["px"][strong], pk["py"][strong], pk["frame"][strong]

    maps = [(name, _pixel_map(name, nx, ny)) for name in _MAP_NAMES]
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
                         clean_tol=0.02, rms_ok=0.01, refine=True, log=None):
    """Fit (detector, UB, R0, osc_axis, sense, phi0, increment) from the tabbin.

    par_path / tab_path : explicit CrysAlisPro par and peak-hunt tabbin; by
        default the par is discovered in `folder` and the tabbin is the one that
        belongs to that par (`find_tabbin(folder, par_path=...)`).
    refine : if True (default), a second fit follows the first fit. It uses
        the sub-pixel centroids of the tabbin if they make the held-out
        residual smaller. It selects the peaks again in each pass
        (`_refine_reselect`), from the strong peaks of the table that are not
        held out. The function keeps the second fit only if it puts the
        check peaks nearer to integer hkl (`_second_fit_is_better`)
        and if the clean peaks pass the self-check with it. Else the fit
        before the second fit stays. False gives the first fit only.

    The fit uses ~80% of the cleanly indexed peaks; the remaining ~20%, spread
    over the scan, validate it (median hkl deviation through the fitted
    geometry, limit `HKL_DEV_LIMIT`). Returns a FittedGeometry. Raises if the
    evidence is weak or no good fit is found (honest failure).

    The first fit selects its peaks one time, with the CrysAlisPro reciprocal
    vectors and the UB of the par. If the par UB or the instrument model of
    CrysAlisPro is not good, this selection keeps only the peaks that agree
    with them, and the fit cannot find the full error of the oscillation axis.
    The second fit removes this limit.
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
        return (_lab_vectors(p, col, row, Fc, Fr, ps, wl)
                - _model_vectors(p, g, dphi, sense)).ravel()

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
    n_fit = int(fit.sum())

    centroids = refined = False
    n_refined, axis_change = 0, 0.0
    if refine:
        def model_resid(q, c, r, gg, dd):
            return (_lab_vectors(q, c, r, Fc, Fr, ps, wl) - _model_vectors(q, gg, dd, sense)).ravel()

        def held_rms(q, c, r):
            return float(np.sqrt(np.mean(model_resid(q, c[held], r[held], g_all[held],
                                                     dphi_all[held]) ** 2)))

        to_pix = _pixel_map(mp_name, nx, ny)
        px_pk, py_pk = pk["px"].astype(float), pk["py"].astype(float)
        col_pk, row_pk = to_pix(px_pk, py_pk)              # all peaks of the table
        # Sub-pixel centroids. Do the first fit again with them. Use them only
        # if the held-out clean peaks then have a smaller residual. A centroid
        # that is not a finite number gives the integer pixel.
        cx = np.where(np.isfinite(pk["cenx"]), pk["cenx"], 0.0)
        cy = np.where(np.isfinite(pk["ceny"]), pk["ceny"], 0.0)
        col_cen, row_cen = to_pix(px_pk + cx, py_pk + cy)
        kc, kr = col_cen[keep], row_cen[keep]
        try:
            p_cen = least_squares(model_resid, p, args=(kc[fit], kr[fit], g, dphi),
                                  loss="soft_l1", f_scale=0.02, max_nfev=250).x
            rms_int, rms_cen = held_rms(p, col_all, row_all), held_rms(p_cen, kc, kr)
            if rms_cen < rms_int:
                centroids, p = True, p_cen
                col_all, row_all, col_pk, row_pk = kc, kr, col_cen, row_cen
                col, row = col_all[fit], row_all[fit]
                rr = float(np.sqrt(np.mean(resid(p, Fc, Fr, sense) ** 2)))
            say(f"sub-pixel centroids: {'used' if centroids else 'not used'}. Held-out rms "
                f"{rms_int:.5f} 1/A without them, {rms_cen:.5f} 1/A with them.")
        except Exception as exc:                           # noqa: BLE001
            say(f"sub-pixel centroids: error in the fit ({type(exc).__name__}: {exc}). Not used.")

        # Second fit, with the strong peaks of the table. It does not use the
        # held-out clean peaks of the first fit and each 5th peak of the table
        # in scan order. The strong peaks among them are the check peaks. Some
        # check peaks were fitted by the first fit: the check thus favours the
        # first fit a little.
        keep_text = "The fit before the second fit stays."
        try:
            dphi_pk = (pk["frame"] - 1) * incr
            held_pk = np.zeros(len(dphi_pk), bool)
            held_pk[np.where(keep)[0][held]] = True
            held_pk[np.argsort(pk["frame"], kind="stable")[4::5]] = True
            strong = _strongest(pk["intensity"], REFINE_STRONG_FRACTION)
            hkl_1 = _hkl_of_peaks(p, col_pk, row_pk, dphi_pk, UB, Fc, Fr, sense, ps, wl)
            n_1 = np.round(hkl_1)
            check = (held_pk & strong & (np.abs(hkl_1 - n_1).max(1) < REFINE_CHECK_TOL)
                     & np.any(n_1 != 0, axis=1))
            p_2, n_2, rr_2 = _refine_reselect(p, col_pk, row_pk, dphi_pk, ~held_pk & strong,
                                              UB, Fc, Fr, sense, ps, wl)
            if p_2 is None or check.sum() < REFINE_MIN_PEAKS:
                say(f"second fit: not enough peaks. {keep_text}")
            else:
                hkl_2 = _hkl_of_peaks(p_2, col_pk[check], row_pk[check], dphi_pk[check],
                                      UB, Fc, Fr, sense, ps, wl)
                dist_1 = np.linalg.norm((UB @ (hkl_1[check] - n_1[check]).T).T, axis=1)
                dist_2 = np.linalg.norm((UB @ (hkl_2 - n_1[check]).T).T, axis=1)
                better, m1, m2, frac = _second_fit_is_better(dist_1, dist_2,
                                                             pk["intensity"][check])
                turn = float(np.degrees(np.arccos(np.clip(
                    _sph(p[6], p[7]) @ _sph(p_2[6], p_2[7]), -1.0, 1.0))))
                say(f"second fit: {n_2} strong peaks in the last pass. {int(check.sum())} strong "
                    f"check peaks: weighted mean distance from integer hkl {m1:.5f} -> {m2:.5f} "
                    f"1/A, smaller in {100 * frac:.1f} % of {REFINE_RESAMPLES} resamples. The "
                    f"rotation axis moved {turn:.4f} deg. "
                    + ("The second fit is better." if better
                       else f"The second fit is not better. {keep_text}"))
                if better:
                    p_before, rr_before = p, rr
                    p, rr, refined, n_refined, axis_change = p_2, rr_2, True, n_2, turn
        except Exception as exc:                           # noqa: BLE001
            say(f"second fit: error ({type(exc).__name__}: {exc}). {keep_text}")

    def geometry(q):
        R0 = _Rot.from_rotvec(q[3:6]).as_matrix()
        osc = _sph(q[6], q[7]); osc = osc/np.linalg.norm(osc)
        det = FlatDetector(distance=q[2], pixel_size=ps, beam_center=(q[0]-1, q[1]-1),
                           shape=(ny, nx), wavelength=wl, beam=_Z, fast=Fc, slow=Fr)
        return det, R0, osc

    # Validate: do the tabbin peaks map to integer hkl through this detector?
    # Both the fitted subset and the peaks the fit never saw must pass.
    from .geometry import _axis_rot

    def validate(q):
        det, R0, osc = geometry(q)

        def hkl_dev(cols, rows, dphis, stride):
            devs = []
            for i in range(0, len(cols), stride):
                rlab = det.scattering_vectors([[cols[i]-1, rows[i]-1]])[0]
                Rn = _axis_rot(tuple(osc), sense*dphis[i]) @ R0
                h = np.linalg.inv(Rn @ UB) @ rlab
                devs.append(np.linalg.norm(h - np.round(h)))
            return float(np.median(devs)) if devs else float("nan")

        return (hkl_dev(col, row, dphi, max(1, len(col)//400)),
                hkl_dev(col_all[held], row_all[held], dphi_all[held], 1))

    def passes(m, mh):
        return m <= HKL_DEV_LIMIT and (not held.any() or mh <= HKL_DEV_LIMIT)

    med, med_held = validate(p)
    if refined and not passes(med, med_held):
        med_0, held_0 = validate(p_before)
        if passes(med_0, held_0):
            say(f"second fit: the clean peaks fail the self-check with it (median dev "
                f"{med:.4f} rlu, held-out {med_held:.4f} rlu). The fit before the second fit stays.")
            p, rr, med, med_held = p_before, rr_before, med_0, held_0
            refined, n_refined, axis_change = False, 0, 0.0
    if refined:
        say(f"second fit: used. The rotation axis was corrected by {axis_change:.4f} deg.")
    det, R0, osc = geometry(p)
    say(f"validation: fitted peaks -> integer hkl, median dev {med:.4f} rlu; "
        f"held-out {int(held.sum())} peaks -> median dev {med_held:.4f} rlu")
    if med > HKL_DEV_LIMIT:
        raise RuntimeError(f"tabbin geometry fit failed self-check (median hkl dev {med:.3f} rlu)")
    if held.any() and med_held > HKL_DEV_LIMIT:
        raise RuntimeError(
            f"tabbin geometry does not generalise: held-out peaks deviate "
            f"{med_held:.3f} rlu from integer hkl (fitted peaks {med:.3f} rlu)")

    return FittedGeometry(det, np.asarray(UB), R0, tuple(float(x) for x in osc),
                          int(sense), float(phi0), float(incr), rr,
                          n_refined if refined else n_fit, med,
                          median_hkl_dev_heldout=med_held, n_heldout=int(held.sum()),
                          mapping_margin=margin, pixel_mapping=mp_name,
                          par_path=str(par), tabbin_path=str(tab),
                          centroids=centroids, refined=refined,
                          n_peaks_refined=int(n_refined), axis_change_deg=axis_change)
