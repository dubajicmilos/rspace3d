"""Per-pixel geometric intensity corrections for reciprocal-space reconstruction.

Convert raw detector counts into an intensity proportional to the scattered
intensity per unit solid angle, correcting the two pixel-dependent geometric
effects on a flat detector:

* solid angle (obliquity): a flat-detector pixel of fixed area A subtends a solid
  angle dOmega = A cos^3(alpha) / L^2, where alpha is the angle between the ray to
  the pixel and the detector normal (the beam). To recover intensity per steradian,
  multiply counts by 1/cos^3(alpha) (= 1 on axis, growing towards the corners).

* polarisation: Thomson scattering of a (partially) polarised beam attenuates the
  scattered intensity by a pixel-dependent factor. For a beam with fraction p
  linearly polarised along the in-plane horizontal unit vector h (and 1-p along
  the vertical v), the factor is
      P = p (1 - (h . khat)^2) + (1 - p) (1 - (v . khat)^2),
  where khat is the outgoing unit direction. Multiply counts by 1/P.

The Lorentz factor is deliberately NOT included. In a rotation-scan reconstruction
that normalises each voxel by its number of contributing pixels, the varying time
a reciprocal-space point spends on the Ewald sphere appears as a varying pixel
count, so dividing by the count already applies the geometric Lorentz correction
(the Meerkat convention).

Polarisation axis. A synchrotron beam is polarised in the orbit plane, i.e.
perpendicular to the (vertical) rotation axis of the diffractometer. The lab frame
of a tabbin-fitted geometry is only defined up to a rotation about the beam, so
neither the detector `fast` axis nor lab +X is physically anchored;
`reconstruct_dataset` therefore passes `horizontal = beam x osc_axis`. Called
directly, `pixel_corrections` falls back to `detector.fast` unless `horizontal`
is given. The factor varies by up to ~2x across the detector between the two
orthogonal choices, so the axis matters and should match the instrument.
"""
from __future__ import annotations

import numpy as np


def pixel_directions(detector, r_lab=None):
    """Outgoing unit direction khat (Npix,3) and obliquity cos(alpha) (Npix,).

    If r_lab (scattering vectors, in the reconstruction's pixel order) is given,
    khat is derived from it so the ordering matches exactly; otherwise it is built
    from the detector pixel grid (row-major over slow, fast).
    """
    if r_lab is None:
        ny, nx = detector.shape
        ys, xs = np.mgrid[0:ny, 0:nx]
        px = np.column_stack([xs.ravel(), ys.ravel()]).astype(float)
        r_lab = detector.scattering_vectors(px)
    # r_lab = s1 - s0 with s0 = beam/lambda, s1 = khat/lambda  =>  khat = lambda*r_lab + beam
    khat = np.asarray(r_lab, float) * detector.wavelength + detector.beam
    khat = khat / np.linalg.norm(khat, axis=1)[:, None]
    cos_alpha = khat @ detector.beam
    return khat, cos_alpha


def pixel_corrections(detector, r_lab=None, *, solid_angle=True, polarization=True,
                      polarization_fraction=0.95, horizontal=None):
    """Per-pixel multiplicative correction (Npix,) float64.

    Ordered to match detector.scattering_vectors over np.mgrid[0:ny, 0:nx].ravel()
    (the reconstruction's pixel order). Returns all-ones if both effects are off.

    Parameters
    ----------
    r_lab : optional (Npix,3) scattering vectors to tie the ordering to the caller.
    solid_angle, polarization : toggle each correction.
    polarization_fraction : fraction of the beam polarised along `horizontal`
        (~0.9-0.95 for a synchrotron; 0.5 = unpolarised).
    horizontal : in-plane unit vector of the dominant polarisation; defaults to
        detector.fast (see the module docstring: prefer beam x oscillation axis).
    """
    khat, cos_alpha = pixel_directions(detector, r_lab)
    corr = np.ones(khat.shape[0], np.float64)
    if solid_angle:
        corr = corr / np.clip(cos_alpha, 1e-9, None) ** 3
    if polarization:
        h = np.asarray(detector.fast if horizontal is None else horizontal, float)
        h = h / np.linalg.norm(h)
        v = np.cross(detector.beam, h)
        v = v / np.linalg.norm(v)
        p = float(polarization_fraction)
        pol = p * (1.0 - (khat @ h) ** 2) + (1.0 - p) * (1.0 - (khat @ v) ** 2)
        corr = corr / np.clip(pol, 1e-9, None)
    return corr
