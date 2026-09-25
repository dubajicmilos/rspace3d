# rspace3d.rawrecon: raw-frame reciprocal-space reconstruction

`rawrecon` reconstructs a 3-D reciprocal-space volume directly from the raw rotation
frames (`.cbf`) of a CrysAlisPro dataset. The reconstruction is GPU-accelerated, and
the geometry calibration is instrument-agnostic: it is based on the CrysAlisPro
peak-hunt table.

`rawrecon` is the counterpart of the `.img`/unwarp path used in the rest of
`rspace3d`. Instead of loading the unwarp layers pre-computed by CrysAlisPro, it maps
every detector pixel of every frame to a fractional `(h,k,l)` and accumulates its
intensity in a voxel grid. The resulting volume goes through the existing
`bin_volume` / `symmetrize_volume` / `save_volume_h5` / viewer code unchanged.

---

## 1. Quick start

### GUI
```
python -m rspace3d.reconstruct_gui        # or the `rspace3d-reconstruct` console script
```
Browse to a dataset folder (raw `.cbf` + `*_cracker.par` [+ `*_peakhunt.tabbin`]),
set the `h/k/l` min/max and the voxel size `dq`, and click **Reconstruct**. Then
click **Symmetrise + Save** or **Open in Viewer**. Leave *"Geometry from CrysAlisPro
tabbin"* ticked; it is the default and the more robust option (§3).

### Programmatic
```python
from rspace3d import rawrecon

# one call: geometry (from tabbin) + reconstruction
vol, info = rawrecon.reconstruct_dataset(
    folder, ranges=((-6, 6), (-6, 6), (-6, 6)), step=0.025,
    corrections=True, use_tabbin=True)

# -> vol is a rawrecon.Volume (NaN = unmeasured). Convert to an rspace3d VolumeData:
vd = rawrecon.reconstruct_to_volumedata(vol, source_folder=folder)
from rspace3d.volume_builder import symmetrize_volume, save_volume_h5
save_volume_h5("recon.h5", symmetrize_volume(vd, "m-3m", sigma=3))
```

---

## 2. What it needs from a dataset folder

| File | Used for | Required |
|------|----------|----------|
| `<stem>_NNNN.cbf` (or `_NNNNN.cbf`) | the raw rotation frames (intensities + goniometer angles) | yes |
| `*_cracker.par` | the UB matrix (cell + orientation) and wavelength | yes |
| `*_peakhunt.tabbin` | CrysAlisPro's indexed peaks, used as the geometry ground truth | for `use_tabbin=True` (recommended) |

Notes:
- Frame discovery (`find_cbf_frames`) does not depend on the folder name: the `.cbf`
  stem need not match it (e.g. folder `12345-pil3cbf-files-sample_01` holds
  `sample_01_00001.cbf`). Numbered files are grouped by prefix, the largest group
  is used, and the number of digits is arbitrary.
- Use the `*_cracker.par`, never a bare `*_0.par`. The plain par often stores a
  reduced cell (e.g. a monoclinic sub-cell) that will not index the frames, whereas
  the cracker par holds the pseudocubic UB that CrysAlisPro actually refined.
  `find_crysalis_par` prefers the cracker par and has a `recursive=True` option
  (GUI: *"Search subfolders for the .par file"*).

---

## 3. Two ways to get the geometry

For each frame, the reconstruction needs the map `hkl = (R_n·UB)⁻¹·r_lab` with
`R_n = R_osc(sense·(φ_n − φ₀); osc_axis)·R0`. The geometry
`(detector pose, UB, R0, osc_axis, sense)` can be obtained in two ways.

### (a) `orient_from_frame`: per-frame auto-indexing (no tabbin)
This path reads the detector geometry from the CBF header and the UB from the par,
detects Bragg peaks on frame 1, and fits `R0` with `index_frame`. The oscillation
axis is taken as lab Y. A quality
gate requires at least 6 detected peaks to lie within 0.1 rlu of integer hkl;
otherwise the function raises an error instead of returning a smeared volume.

This path works only when three conditions hold:
1. the CBF-header geometry (distance, beam centre) is correct;
2. the detector readout matches the default (fast = +x, slow = +y);
3. the frame has clean single-crystal peaks (no powder), so that indexing succeeds.

### (b) `geometry_from_tabbin`: instrument-agnostic fit (recommended)
This path fits the entire geometry to the CrysAlisPro peak-hunt table. Nothing
instrument-specific is assumed: the header is used only as a starting guess and may
be wrong. The path is intended for any flat-detector, single-axis, monochromatic
dataset (§4).

`reconstruct_dataset(..., use_tabbin=True)` uses (b); `use_tabbin=False` uses (a).

---

## 4. Instrument-agnostic calibration (`geometry_from_tabbin`)

For each indexed reflection, the tabbin gives a correspondence
(pixel px,py) ↔ (frame) ↔ (reciprocal vector dx,dy,dz). We fit the geometry to
thousands of these correspondences by least squares:

1. Read the par to obtain `UB` (columns a*,b*,c* in 1/d) and the wavelength `λ`.
2. Read the tabbin (`read_tabbin`) to obtain `px, py, frame, (dx,dy,dz)` for each
   peak. Recover `hkl = (UB·λ)⁻¹·(dx,dy,dz)`; the tabbin stores the crystal-frame
   reciprocal vector scaled by the wavelength, `dxdydz = λ·UB·hkl`. Keep the peaks
   whose hkl lie within `clean_tol` (default 0.02) of integers.
3. Ground the pixel mapping (`_ground_mapping`, §4.1): determine how the tabbin
   `px,py` map to the `(col,row)` grid of the CBF frames.
4. Fit `(beam-centre, distance, R0, oscillation-axis, sense)` with
   `scipy.optimize.least_squares` (robust `soft_l1` loss), minimising the residual
   `r_lab(pixel; detector) − R_osc(Δφ; osc_axis, sense)·R0·(UB·hkl)`. The fit is
   tried for the 8 in-plane detector orientations × 2 senses × oscillation-axis
   seeds {Y, X, Z}, and the solution with the lowest residual is kept.
5. Self-validate: map the tabbin peaks back to hkl through the fitted detector and
   check that they land on integers. The median deviation is reported, and the
   function raises an error if it exceeds 0.1 rlu.

The function returns a
`FittedGeometry(detector, UB, R0, osc_axis, sense, phi0, increment, rms,
n_peaks, median_hkl_dev)`.

### 4.1 The pixel-mapping ambiguity and how it is resolved

The tabbin peak positions alone cannot determine how CrysAlisPro's `(px,py)` indexing
maps to the `(col,row)` grid of the CBF frames; the two can differ by a transpose
and/or an axis flip. Every such option gives a self-consistent fit (the residual is
low and the integer-hkl self-check passes), because the fit compensates through `R0`
and the beam centre. Only one option, however, matches the physical detector that the
reconstruction reads. With the wrong option, the reconstruction is smeared
(integer-Bragg RMS ≈ 0.5 rlu) even though every internal check has passed. This
error is easy to make with a near-square detector (Eiger 2068×2162) or a flipped
readout (Pilatus).

`_ground_mapping` resolves the ambiguity with the measured data. For the strongest
peaks, it tests all 8 transpose/flip options against the actual CBF-frame intensities
and keeps the mapping under which the peaks land on bright pixels of the frame. On
both test datasets (§7) the grounding selects the mapping `TXY` (transpose + x-flip +
y-flip) relative to fabio's `(row,col)`.

---

## 5. The `.tabbin` format (decoded)

CrysAlisPro peak-hunt tables (`*_peakhunt.tabbin`) are binary and little-endian. The
layout below was verified on version 3. Reference implementation:
`github.com/DESY-Petra-III/CrysalisTabbiner`.

```
Header (312 bytes):
  uint32 @0   number of peaks
  uint16 @10  version (2..16)

Then N records of 168 bytes each:
  float64 @0,8,16   dx, dy, dz   reciprocal vector, CRYSTAL frame, wavelength-scaled
                                 (dxdydz = lambda * UB_1d * hkl;  |d| = lambda / d-spacing)
  float64 @24       dlength      = |d|
  uint32  @32       intensity
  uint16  @44,46    px, py       detector pixel, 0-based (add 1 for the CrysAlis GUI)
  float32 @48,52    centroid x,y (sub-pixel offset)
  int16   @160      frame        1-based frame number
```
`rawrecon.read_tabbin(path)` returns a dict of numpy arrays, with `px`,`py` converted
to 1-based. It validates the file size and version and raises an error on an
unrecognised layout, so a format change cannot be misread silently.

Miller indices are recovered as `hkl = inv(UB * wavelength) @ [dx,dy,dz]`, where `UB`
is the 1/d matrix from `read_crysalis_par` (the raw par UB divided by the wavelength).

---

## 6. Practical notes

- Header geometry can be wrong. On the I15 test dataset (PILATUS3 2M) the CBF
  header gives a distance of 370 mm, whereas the geometry that fits the
  diffraction has an effective distance of 649 mm. |Q| constrains only the ratio
  `pixel_size / distance`, and 649 mm at 172 µm is equivalent to 370 mm at 98 µm.
  A reconstruction with the header value is unusable, so the tabbin fit uses the
  header only as a starting guess.
- The detector readout can be transposed or flipped relative to the Eiger
  convention. The orientation search and the intensity grounding (§4.1) handle
  this.
- Powder rings from the sample environment can dominate the peaks found on a
  single frame, and simple threshold-and-centroid peak finding cannot separate
  them from the single-crystal spots; `orient_from_frame` then rejects the frame
  through its quality gate. CrysAlisPro's 3-D multi-frame peak hunt does separate
  them, which makes the tabbin path the more robust route.
- Strong Bragg peaks can be saturated. Do not clip them during peak finding, and
  set the reconstruction `hot` clip above the detector's real maximum or leave it
  at `None`.
- Powder rings (bright, at non-integer positions) can mislead
  `bragg_registration_rms`, which can then read about 0.5 rlu on a correct
  reconstruction. On powder-contaminated data, check the tabbin-peak to
  integer-hkl deviation and inspect the volume by eye rather than relying on this
  metric alone.

---

## 7. Validation

Both datasets are reconstructed by the same `geometry_from_tabbin` code, with no
per-instrument branches:

| Dataset | Detector | Fitted distance | Fitted beam centre | Fit rms | Bragg RMS |
|---------|----------|-----------------|--------------------|---------|-----------|
| I19-2 (MAPbBr3 215 K) | Eiger 2X 4M | 85.3 mm | (989.2, 1422.8) | 0.0012 1/Å | 0.033 rlu |
| I15 (MAPbI3 340 K) | PILATUS3 2M | 649.2 mm | (759.6, 853.7) | 0.0014 1/Å | 0.034 rlu |

I19-2 has two independent sources of geometry, which allows a cross-check. The tabbin
fit gives a beam centre of (989.2, 1422.8), within one pixel of the CBF-header value
(990.0, 1423.3), which is known to be correct. The fitted distance is 85.3 mm, against
85.0 mm in the header. The reconstructions from the tabbin geometry and from the
header with auto-indexing correlate at 0.83 on the L=0 plane, so the tabbin path
agrees with the independent route.

---

## 8. API reference

| Symbol | Module | Purpose |
|--------|--------|---------|
| `reconstruct_dataset(folder, *, ranges, step, use_tabbin, corrections, hot, use_gpu, nframes, ...)` | `engine` | one-call: geometry + reconstruction → `(Volume, info)` |
| `geometry_from_tabbin(folder) -> FittedGeometry` | `calibrate` | instrument-agnostic geometry fit from the tabbin |
| `orient_from_frame(par, frame1) -> (det, UB, R0, phi0, idx)` | `geometry` | per-frame auto-index (with quality gate) |
| `reconstruct_volume_gpu / reconstruct_volume(frames, phis, UB, R0, det, *, phi0, osc_axis, sense, ranges, step, hot, corrections, ...)` | `engine` | the core GPU / CPU histogram engine |
| `reconstruct_to_volumedata(vol) -> VolumeData` | `engine` | wrap as an rspace3d `VolumeData` (NaN = unmeasured) |
| `read_tabbin(path)`, `find_tabbin(folder)` | `tabbin` | read / locate the peak-hunt table |
| `find_cbf_frames(folder)`, `find_crysalis_par(folder, recursive=)` | `geometry` | robust file discovery |
| `pixel_corrections(det, r_lab, *, solid_angle, polarization, polarization_fraction, horizontal)` | `corrections` | per-pixel geometric corrections |
| `bragg_registration_rms(vol)` | `engine` | reference-free integer-registration check (see caveat §6) |

`Volume`: `data (nh,nk,nl; NaN=unmeasured)`, `H,K,L` axes, `UB` (1/d), `counts`,
`wavelength`. The grid may be anisotropic (independent min/max per axis, common `dq`).

---

## 9. Output format and the measured/unmeasured distinction

A volume wrapped by `reconstruct_to_volumedata` and saved with `save_volume_h5` uses
the rspace3d HDF5 layout (`/data`, `/H`, `/K`, `/L`, `/UB`, `/M_inv`, + `wavelength`,
`cell_*`, `s` attrs), so it opens directly in `rsp_viewer` and can be passed to
`bin_volume` and `symmetrize_volume`.

Because the reconstruction records the exact number of pixels that contribute to each
voxel, the measured/unmeasured distinction is exact and needs no coverage-mask
heuristic:
- an unmeasured voxel (no contributing pixels) is NaN;
- a measured voxel with zero intensity is 0.0.

`symmetrize_volume` masks on `isfinite`, so unmeasured (NaN) voxels are excluded and
measured zeros are kept.

> Viewer compatibility: the h5 file must carry a non-zero `wavelength` attribute and
> the UB in CrysAlisPro's λ-scaled convention (`UB_stored = UB_1d · λ`). Without the
> wavelength, the viewer computes `UB/wavelength → NaN` and shows an empty plane.
> `reconstruct_to_volumedata` followed by `save_volume_h5` takes care of both.

---

## 10. Assumptions and limitations

The calibration is instrument-agnostic only within the standard rotation experiment,
which means:
- a flat area detector (a curved detector would need a different projection);
- a single-axis rotation scan (φ or ω; a multi-axis scan would need per-frame
  orientations rather than one oscillation axis);
- a monochromatic beam.

Other caveats:
- The calibration needs the tabbin and enough cleanly indexed peaks spread over
  enough frames (I19-2: 2563, I15: 1755). A very sparse scan could leave the fit
  under-constrained.
- The tabbin binary layout can vary between CrysAlisPro versions; `read_tabbin`
  validates the layout and raises an error on a mismatch rather than misreading the
  file.
- For the polarisation correction, the E-vector is taken perpendicular to the beam and
  to the fitted oscillation axis (`beam x osc_axis`, i.e. horizontal for a vertical
  rotation axis). The lab frame of the tabbin fit is defined only up to a rotation
  about the beam, so neither the detector fast axis nor lab +X is physically anchored,
  and the polarisation factors for two orthogonal E-vector directions differ by up to
  ~2x at the detector edges. For an instrument with a horizontal rotation axis,
  override the E-vector with `corrections={'horizontal': vector}`.

---

## 11. Module map

```
rawrecon/
  geometry.py    FlatDetector, detect_peaks, index_frame, _axis_rot,
                 find_cbf_frames, find_crysalis_par, read_crysalis_par,
                 orient_from_frame (per-frame auto-index + quality gate)
  corrections.py pixel_corrections (solid-angle + polarisation; Lorentz via
                 count-normalisation, the Meerkat convention)
  engine.py      Volume, reconstruct_volume[_gpu], reconstruct_dataset,
                 reconstruct_to_volumedata, bragg_registration_rms, has_cupy
  tabbin.py      read_tabbin, find_tabbin
  calibrate.py   geometry_from_tabbin, FittedGeometry, _ground_mapping
```
