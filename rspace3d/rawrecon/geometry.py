"""Detector geometry, CrysAlisPro par reading, and frame indexing.

The reconstruction-relevant subset of the xrays_on_detector `realframe` module,
kept isolated inside rspace3d.rawrecon so the raw-CBF reconstruction path does not
entangle with the existing .img/unwarp code.

Lab frame: beam +z (source -> sample -> detector), detector fast axis +x, slow
axis +y, detector plane at z = distance. Units follow the crystallographic 1/d
convention (a CrysAlisPro UB divided by wavelength gives a*,b*,c* with |a*| = 1/a).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FlatDetector:
    """Flat area detector on the beam axis (2theta = 0)."""
    distance: float                 # sample -> detector (same unit as pixel_size)
    pixel_size: float
    beam_center: tuple              # (fast_x, slow_y) in pixels
    shape: tuple                    # (n_slow, n_fast)
    wavelength: float               # Angstrom
    beam: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    fast: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0]))
    slow: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0, 0.0]))

    @classmethod
    def from_eiger_cbf(cls, path: str):
        """Build geometry + read the goniometer angles from a Dectris miniCBF.

        Returns (detector, angles, image) where angles is a dict of the header
        goniometer values (deg) and image is the int32 pixel array.
        """
        import fabio

        img = fabio.open(path)
        hdr = img.header.get("_array_data.header_contents", "")
        if not hdr:                                   # fall back to raw text
            with open(path, "rb") as f:
                hdr = f.read(4096).decode("latin-1", "replace")

        def num(key):
            import re
            m = re.search(rf"#\s*{re.escape(key)}\s+\(?\s*([-\d.eE]+)", hdr)
            return float(m.group(1)) if m else None

        def pair(key):
            import re
            m = re.search(rf"#\s*{re.escape(key)}\s*\(\s*([-\d.eE]+)\s*,\s*([-\d.eE]+)",
                          hdr)
            return (float(m.group(1)), float(m.group(2))) if m else None

        dist_m = num("Detector_distance")
        pix_m = num("Pixel_size")
        bc = pair("Beam_xy")
        lam = num("Wavelength")
        ny, nx = img.data.shape
        det = cls(distance=dist_m * 1e3, pixel_size=pix_m * 1e3,
                  beam_center=bc, shape=(ny, nx), wavelength=lam)
        angles = {a: num(a) for a in ("Phi", "Kappa", "Omega", "Start_angle",
                                      "Angle_increment", "Detector_2theta")}
        return det, angles, img.data.astype(np.int32)

    def scattering_vectors(self, px: np.ndarray) -> np.ndarray:
        """Pixel (fast, slow) -> scattering vector r = s1 - s0 (1/Angstrom)."""
        px = np.atleast_2d(np.asarray(px, float))
        u = (px[:, 0] - self.beam_center[0]) * self.pixel_size
        v = (px[:, 1] - self.beam_center[1]) * self.pixel_size
        P = (u[:, None] * self.fast + v[:, None] * self.slow
             + self.distance * self.beam)
        s1 = P / np.linalg.norm(P, axis=1)[:, None] / self.wavelength
        return s1 - self.beam / self.wavelength


def detect_peaks(image, beam_center, thr=200, rmin=170, smin=3, smax=200):
    """Label pixels above `thr`, return spot centroids (fast_x, slow_y)."""
    from scipy import ndimage

    lbl, n = ndimage.label(image > thr)
    sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
    coms = ndimage.center_of_mass(image, lbl, range(1, n + 1))
    out = []
    for c, s in zip(coms, sizes):
        x, y = c[1], c[0]
        if smin <= s <= smax and np.hypot(x - beam_center[0], y - beam_center[1]) > rmin:
            out.append((x, y))
    return np.array(out)


def _kabsch(A, B):
    U, _, Vt = np.linalg.svd(A @ B.T)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    return Vt.T @ np.diag([1, 1, d]) @ U.T


@dataclass
class IndexResult:
    R: np.ndarray            # crystal-to-lab rotation for the frame
    hkl: np.ndarray          # (N,3) int, per input peak
    inliers: np.ndarray      # (N,) bool
    rms: float               # 1/Angstrom


def index_frame(peaks, UB, detector: FlatDetector,
                tol=0.012, dtol=0.02, atol_deg=1.5, min_inliers=5,
                hkl_max=9) -> IndexResult | None:
    """Known-cell pair indexing: find R with r_obs = R @ (UB @ hkl).

    UB columns are a*,b*,c* in 1/d units (i.e. CrysAlis UB / lambda). Only the
    cell (|a*|, angles) needs to be right - the orientation is fitted here."""
    r_obs = detector.scattering_vectors(peaks)
    UBinv = np.linalg.inv(UB)
    hs = np.arange(-hkl_max, hkl_max + 1)
    HKL = np.stack(np.meshgrid(hs, hs, hs, indexing="ij"), -1).reshape(-1, 3)
    HKL = HKL[np.any(HKL != 0, 1)]
    G = (UB @ HKL.T).T
    GN = np.linalg.norm(G, axis=1)

    vn = np.linalg.norm(r_obs, axis=1)
    order = np.argsort(vn)
    i0 = order[0]
    cand0 = np.nonzero(np.abs(GN - vn[i0]) < dtol)[0]
    best = None
    for i1 in order[1:]:
        cand1 = np.nonzero(np.abs(GN - vn[i1]) < dtol)[0]
        ang_obs = np.degrees(np.arccos(np.clip(
            r_obs[i0] @ r_obs[i1] / (vn[i0] * vn[i1]), -1, 1)))
        for a in cand0:
            for b in cand1:
                ang = np.degrees(np.arccos(np.clip(
                    G[a] @ G[b] / (GN[a] * GN[b]), -1, 1)))
                if abs(ang - ang_obs) > atol_deg:
                    continue
                R = _kabsch(np.stack([G[a], G[b]], 1),
                            np.stack([r_obs[i0], r_obs[i1]], 1))
                hkl = np.round((UBinv @ R.T @ r_obs.T).T)
                inl = np.linalg.norm((R @ (UB @ hkl.T)).T - r_obs, axis=1) < tol
                if inl.sum() < min_inliers:
                    continue
                R = _kabsch(UB @ hkl[inl].T, r_obs[inl].T)
                resid = np.linalg.norm((R @ (UB @ hkl.T)).T - r_obs, axis=1)
                inl = resid < tol
                rms = float(np.sqrt(np.mean(resid[inl] ** 2)))
                score = (int(inl.sum()), -rms)
                if best is None or score > best[0]:
                    best = (score, IndexResult(R, hkl.astype(int), inl, rms))
    return best[1] if best else None


def _axis_rot(axis, deg):
    axis = np.asarray(axis, float); axis = axis / np.linalg.norm(axis)
    t = np.radians(deg); c, s = np.cos(t), np.sin(t)
    x, y, z = axis
    K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) * c + s * K + (1 - c) * np.outer(axis, axis)


def find_crysalis_par(folder, recursive=False):
    """Locate the CrysAlisPro .par carrying the refined UB.

    Prefers `*_cracker.par` (the refined orientation matrix) over a bare `*.par`:
    the latter can hold an automatic *reduced* cell (e.g. a monoclinic sub-cell)
    that will NOT index the pseudocubic perovskite frames, whereas the cracker par
    holds the pseudocubic UB actually used for the reconstruction.

    recursive : also search subfolders of `folder` (one common layout keeps the
        par in a subfolder). Among matches, the one closest to `folder` wins.
    """
    import glob
    import os

    def hits(pat):
        if recursive:
            found = glob.glob(os.path.join(folder, "**", pat), recursive=True)
        else:
            found = glob.glob(os.path.join(folder, pat))
        # prefer shallowest (closest to `folder`), then alphabetical for stability
        return sorted(found, key=lambda p: (p.count(os.sep), p))

    cracker = hits("*_cracker.par")
    if cracker:
        return cracker[0]
    plain = [p for p in hits("*.par") if not p.endswith("_cracker.par")]
    return plain[0] if plain else None


def find_cbf_frames(folder):
    """Discover the main run of numbered .cbf frames in `folder`.

    Returns (sorted_frame_paths, frame_numbers). Robust to naming: the CBF stem is
    NOT assumed to match the folder name (e.g. folder
    '12345-pil3cbf-files-sample_01' holds 'sample_01_00001.cbf').
    Files matching '<prefix><number>.cbf' are grouped by prefix; the largest group
    (the main scan) is returned, sorted by the trailing number. Digit count is
    arbitrary (0001 or 00001). Only the top level is scanned (sub-run folders such
    as bup/ tmp/ are ignored). Returns ([], []) if none are found.
    """
    import glob
    import os
    import re

    names = {os.path.basename(p)
             for p in glob.glob(os.path.join(folder, "*.cbf"))
             + glob.glob(os.path.join(folder, "*.CBF"))}
    pat = re.compile(r"^(.*?)(\d+)\.cbf$", re.IGNORECASE)
    groups: dict[str, list] = {}
    for f in names:
        m = pat.match(f)
        if m:
            groups.setdefault(m.group(1), []).append((int(m.group(2)), f))
    if not groups:
        return [], []
    prefix = max(groups, key=lambda k: len(groups[k]))       # main scan
    items = sorted(groups[prefix])
    paths = [os.path.join(folder, f) for _, f in items]
    nums = [n for n, _ in items]
    return paths, nums


def read_crysalis_par(par_path):
    """Parse (UB, wavelength) from a CrysAlisPro .par file.

    UB is returned as columns a*,b*,c* in the 1/d convention (|column| = 1/d): the
    `CRYSTALLOGRAPHY UB` matrix (row-major 3x3, lambda-scaled) divided by the
    `CRYSTALLOGRAPHY WAVELENGTH`. Use find_crysalis_par to pick the right file.
    """
    import re

    with open(par_path, "r", encoding="latin-1") as f:
        txt = f.read()
    m = re.search(r"CRYSTALLOGRAPHY UB[ \t]+([-\d][^\r\n]*)", txt)
    if not m:
        raise ValueError(f"no 'CRYSTALLOGRAPHY UB' line in {par_path}")
    ub = np.array([float(x) for x in m.group(1).split()[:9]]).reshape(3, 3)
    wm = re.search(r"CRYSTALLOGRAPHY WAVELENGTH[ \t]+([-\d.eE+]+)", txt)
    if wm is None:
        raise ValueError(f"no 'CRYSTALLOGRAPHY WAVELENGTH' line in {par_path}")
    wl = float(wm.group(1))
    return ub / wl, wl


# (detect_peaks kwargs, index_frame kwargs) tried in order. The first setting is
# tuned for a close Eiger (I19-2: 85 mm, 0.49 A, 75 um px -> sharp spots, rms
# ~0.003); the second is a coarser fallback for a far detector with big pixels /
# module gaps (I15 Pilatus: 370 mm, 0.17 A, 172 um px -> rms ~0.02). Tight is tried
# first so the well-behaved case is unaffected.
_ORIENT_SCHEDULE = [
    (dict(thr=200, rmin=170, smin=3, smax=200), dict(tol=0.012, dtol=0.02, atol_deg=1.5)),
    (dict(thr=300, rmin=50, smin=2, smax=300), dict(tol=0.030, dtol=0.04, atol_deg=3.0)),
]


def orient_from_frame(par_path, frame1_path, *, schedule=None, min_registered=6):
    """Full geometry + orientation setup from a CrysAlisPro par + the first frame.

    Returns (detector, UB, R0, phi0, index_result):
      * detector, phi0 and the goniometer datum come from the CBF header;
      * UB (1/d) and the wavelength come from the par (read_crysalis_par);
      * R0 (the rotation linking the UB frame to the detector lab frame - the
        "missing rotation") is found by indexing frame 1.

    Indexing tries a schedule of peak-detection + tolerance settings (see
    _ORIENT_SCHEDULE) so it works across instruments (a close sharp-spot Eiger and
    a far, big-pixel Pilatus need different tolerances). A QUALITY GATE then checks
    that R0 actually places at least `min_registered` detected peaks near integer
    hkl (a genuine orientation does; a loose-tolerance coincidence does not) and
    raises otherwise - better an honest failure than a silently smeared volume.

    R0 is derived per dataset on purpose: across mounts it differs by a lattice-
    symmetry branch, so it is NOT safely reused from another crystal, whereas the
    pixel->hkl map and cell are shared.
    """
    import os

    det, ang, img = FlatDetector.from_eiger_cbf(frame1_path)
    UB, _wl = read_crysalis_par(par_path)
    UBinv = np.linalg.inv(UB)
    best_res, best_nreg = None, -1
    for dp_kw, idx_kw in (schedule or _ORIENT_SCHEDULE):
        peaks = detect_peaks(img, det.beam_center, **dp_kw)
        if len(peaks) < 5:
            continue
        res = index_frame(peaks, UB, det, **idx_kw)
        if res is None:
            continue
        hklf = (UBinv @ res.R.T @ det.scattering_vectors(peaks).T).T
        nreg = int((np.linalg.norm(hklf - np.round(hklf), axis=1) < 0.1).sum())
        if nreg > best_nreg:
            best_res, best_nreg = res, nreg
        if nreg >= min_registered:
            break
    if best_res is None or best_nreg < min_registered:
        raise RuntimeError(
            f"could not reliably index {os.path.basename(frame1_path)}: only "
            f"{max(best_nreg, 0)} detected peaks land near integer hkl (need "
            f">= {min_registered}). The frame's peak list is likely dominated by "
            f"powder / parasitic scatter, or the CBF-header beam centre disagrees "
            f"with the value CrysAlisPro refined. This dataset needs "
            f"CrysAlisPro-grade indexing to obtain the orientation.")
    return det, UB, best_res.R, ang["Phi"], best_res
