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
| `reconstruct_dataset(folder, *, ranges, step, bins, use_tabbin, corrections, hot, use_gpu, nframes, ...)` | `engine` | one-call: geometry + reconstruction → `(Volume, info)`. Voxel centres are multiples of `step` × `bins`, so hkl = 0 is a voxel centre after `bin_volume(vd, *bins)` and all Laue groups stay exact |
| `geometry_from_tabbin(folder) -> FittedGeometry` | `calibrate` | instrument-agnostic geometry fit from the tabbin |
| `orient_from_frame(par, frame1) -> (det, UB, R0, phi0, idx)` | `geometry` | per-frame auto-index (with quality gate) |
| `reconstruct_volume_gpu / reconstruct_volume(frames, phis, UB, R0, det, *, phi0, osc_axis, sense, ranges, step, bins, hot, corrections, ...)` | `engine` | the core GPU / CPU histogram engine |
| `reconstruct_to_volumedata(vol) -> VolumeData` | `engine` | wrap as an rspace3d `VolumeData` (NaN = unmeasured) |
| `read_tabbin(path)`, `find_tabbin(folder)` | `tabbin` | read / locate the peak-hunt table |
| `find_cbf_frames(folder)`, `find_crysalis_par(folder, recursive=)` | `geometry` | robust file discovery |
| `pixel_corrections(det, r_lab, *, solid_angle, polarization, polarization_fraction, horizontal)` | `corrections` | per-pixel geometric corrections |
| `bragg_registration_rms(vol)` | `engine` | reference-free integer-registration check (see caveat §6) |
| `prepare_setup(folder, ...) -> ReconSetup` | `engine` | finds the frames and the geometry without a pass over the frames. `reconstruct_dataset(..., keep_setup=True)` puts the setup in `info['setup']` |
| `ReconSetup.reconstruct(*, positions, groups, frame_fn, use_gpu, progress_callback, cancelled)` | `engine` | one more engine pass with the same geometry |
| `reconstruct_dataset(..., halves=True)` | `engine` | the same pass also makes the odd-frame and even-frame volumes (`vol.parts`). The frames must have odd and even numbers. The host needs about 120 bytes per voxel at the peak, against about 55 bytes per voxel without halves |
| `gpu_memory_bytes(n_voxels, halves=False) -> int` | `engine` | GPU accumulator memory of one pass in bytes (20 bytes per voxel, two times with halves) |
| `ring_statistics(setup) -> RingStats`, `detect_rings(stats) -> RingDetection` | `rings` | powder-ring detector on the frame data |
| `RingMask.from_detection(polar, det)`, `RingMask.from_bands(setup, bands, unit)` | `rings` | per-frame ring-band pixel mask (detected, or bands that you give in px, 1/d or d) |
| `predict_frame(setup, model, i) -> image` | `frames` | predicted raw counts of one frame from a model volume |
| `leave_one_out_median(vol, laue) -> array` | `frames` | median of the other orbit members of each voxel (the prediction of the frame tools) |
| `frame_statistics(setup, model, *, mask) -> FrameStats` | `frames` | pass A: residual statistics per frame, 5-degree sector and 1-pixel radial bin |
| `frame_map(setup, volume, laue)` | `frames` | observed / predicted per frame and sector (`FrameMap.normalised()`: divided by the sector median over the frames), radial residual profile |
| `frame_map_from_stats(setup, stats, rings) -> FrameMap` | `frames` | the frame map from the sums of pass A |
| `fit_frame_model(D, P, N) -> FrameModel` | `frame_model` | the additive table B and the scale table S = 1 + a of each frame |
| `frame_function(cells, model, mask)` | `frames` | the `frame_fn` of the engine for the ring mask and the frame model |
| `corrected_reconstruction(setup, *, model, mask, groups)` | `frames` | reconstruction of g (obs - B) / S with the ring mask |
| `run_frame_chain(setup, laue, *, rings, frame_model, iterations, halves, raw, options, use_gpu, keep_tables, keep_masked, progress, cancelled, log) -> FrameChainResult` | `frames` | the frame chain (checked for mmm): ring mask, frame model (3 iterations), plain orbit mean. `keep_tables=True` keeps the fitted tables in `.model`. `keep_masked=False` drops the ring-mask volume (`.masked`, with the frame model) and its halves (`.masked_halves`) |

`Volume`: `data (nh,nk,nl; NaN=unmeasured)`, `H,K,L` axes, `UB` (1/d), `counts`,
`wavelength`, `variance (nh,nk,nl; NaN=unmeasured)`, `parts` (one `Volume` per frame
group, or None). The grid may be anisotropic
(independent min/max per axis, common `dq`).

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

### Variance and `/uncertainty`

The engine also calculates the variance (σ²) of each voxel value. A voxel value is
the mean of the N pixels that contribute to it:

    I_voxel = Σ_p g_p r_p / N

Here r_p is the raw count of pixel p and g_p is its correction factor (solid angle
and polarisation, 1 without corrections). The raw counts are Poisson. Thus the
variance of the voxel value is:

    σ² = Σ_p g_p² max(r_p, 1) / N²

The floor of 1 count per pixel is the same floor as the Poisson floor of the outlier
rejection. A voxel with N = 0 has a NaN variance. The engine keeps a third float64
accumulator for Σ g² max(r, 1). The accumulation thus needs 20 bytes per voxel
(float64 sum, int32 count, float64 sum of squares).

`reconstruct_to_volumedata` puts the variance in `VolumeData.variance` and sets
`metadata['variance_model'] = 'poisson_counts'`. `save_volume_h5` writes the
dataset `/uncertainty` = √variance (float32, NaN = unmeasured). `load_volume_h5`
reads it back as `variance = uncertainty²`. The attribute `sigma` is a different
value: the outlier threshold of the symmetrisation. `bin_volume` weights each voxel
with its pixel count.
Thus a binned value is equal to the value of a reconstruction with the larger
voxel, and the binned variance is Σ N_i² σ_i² / (Σ N_i)².

The unwarp route (`load_unwarp_folder`) uses a different model,
`'poisson_stored_units'`: the stored `.img` values are treated as Poisson counts,
with the variance max(I, 1) per pixel.

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

Limits of the frame tools (section 12):
- The frame chain (section 12) was validated with Laue group mmm and
  0.2-degree frames (I19-2 and I15 data). Other Laue groups and frame widths
  work, but they are not validated.
- Other Laue groups were not checked on real data. The prediction uses the
  orbit members of the Laue group without the identity. A test on synthetic
  data gave a smaller error for 2/m, 4/mmm and m-3m. For Laue group -1 the
  same test showed no gain from the chain.
- Other frame widths were not checked. The method scales its frame windows
  with 0.2 / width.
- Other detectors and grids that are not symmetric about 0 were not checked.

---

## 11. Module map

```
rawrecon/
  geometry.py    FlatDetector, detect_peaks, index_frame, _axis_rot,
                 find_cbf_frames, find_crysalis_par, read_crysalis_par,
                 orient_from_frame (per-frame auto-index + quality gate)
  corrections.py pixel_corrections (solid-angle + polarisation; Lorentz via
                 count-normalisation)
  engine.py      Volume, reconstruct_volume[_gpu], reconstruct_dataset,
                 reconstruct_to_volumedata, bragg_registration_rms, has_cupy,
                 ReconSetup, prepare_setup, gpu_memory_bytes
  tabbin.py      read_tabbin, find_tabbin
  calibrate.py   geometry_from_tabbin, FittedGeometry, _ground_mapping
  rings.py       PixelPolar, ring_statistics, detect_rings, RingMask
  frame_model.py ModelOptions, fit_frame_model, FrameModel, interp_B, interp_S
  frames.py      FrameCells, predict_frame, leave_one_out_median,
                 frame_statistics, frame_map, frame_map_from_stats,
                 frame_function, corrected_reconstruction, run_frame_chain,
                 FrameStats, FrameMap, FrameChainResult
```

## 12. Frame tools

The frame tools use the frames again after the reconstruction. They find
contamination that changes from frame to frame, for example powder rings of
a wire or a sample holder that is in the beam for some frames, an additive
background in some frames, or a shadow. They need the `ReconSetup` of the
dataset (`reconstruct_dataset(..., keep_setup=True)` puts it in
`info['setup']`), so no new geometry fit is necessary. `prepare_setup` also
gives the setup. `info['setup']` is not a JSON value. Remove it before you
serialise `info`, for example `{k: v for k, v in info.items() if k != 'setup'}`.

The frame chain is: ring mask, frame model (3 iterations), plain orbit mean
(`run_frame_chain`).

```python
from rspace3d import rawrecon
from rspace3d.volume_builder import save_volume_h5

vol, info = rawrecon.reconstruct_dataset(folder, use_tabbin=True, halves=True,
                                         keep_setup=True)
setup = info['setup']
# Use the Laue group of your sample. Only mmm was checked on real data.
res = rawrecon.run_frame_chain(setup, 'mmm', raw=vol, halves=True, log=print)
save_volume_h5('sample_frame_tools.h5', res.volume)   # metadata: frame_tools, frame_correction
```

Only Laue group mmm was checked on real data (see the limits at the end of
section 10).

The chain has these steps:

1. Ring detection. One pass measures the trimmed mean of each 1-pixel annulus
   and of each 30-degree sector of it in each frame. Narrow peaks at 4 or more
   radii in a frame are powder rings if they have the same time course and
   show in many azimuth sectors. A ring must show in at least 5 of the 12
   sectors, or in 60 % of the valid sectors if that is fewer (minimum 2).
   The mask covers each ring band in the frames of its event. The time
   background is a running median over 201 frames (0.2-degree frames). Thus
   the detector needs frames without the rings before or after the ring event
   in the given frames. If all frames have the rings, it finds no ring. The
   ring mask depends on the frames that you give. Check the number of ring
   frames when you change the frame range.
2. The volume with the ring mask. A masked pixel is not used.
3. Three iterations of the frame model. The prediction is the median of the
   other orbit members of each voxel. Pass A compares each frame with the
   prediction in 5-degree sectors and 1-pixel radial bins. The fit finds a
   scale S = 1 + a for each frame and sector and an additive table B for each
   frame, sector and radial bin. The thresholds come from the scatter of the
   data over the frames, so clean frames get no correction. The corrected
   volume uses g (obs - B) / S for each pixel. Each iteration fits the tables
   again from the original frames.
4. The plain orbit mean (equal weights, no rejection) in the Laue group that
   you give.

The variance of a corrected pixel is g² max(obs, 1) / S² (Poisson variance of
the raw count). The error of the model is not in the variance.

`frame_map(setup, volume, laue)` gives the agreement of each frame and sector
with a first average and the radial residual profile. `FrameMap.excess` keeps
the observed / predicted sums of each frame and sector. The display value
(`FrameMap.normalised()`) is this ratio divided by its median over the frames
of the same sector: 1 means agreement with the other frames. `summary()`
counts the frames whose median normalised ratio over the sectors differs from
1 by more than 10 %. A cell without observed counts gets the value 0.001, so
an empty frame shows at the bottom of the scale and counts as off. On real
data the ratio before the normalisation is not 1. For the full frame map of
one I19-2 scan (MAPbBr3) its median was 0.94. Each effect has a direction.
Real changes between frames and sectors lower the ratio. They were the main
cause of the 0.94. The cause of the slow change in this scan is not known.
Absorption or a shadow are possible causes, not tested. The orbit median is
lower than the orbit mean. This raises the ratio (about +2.5 % in a synthetic
test). Cells at Bragg nodes have a low median ratio and a long upper tail,
because a peak is in only a few frames. The fit of the frame model first removes the median over
the frames of the residual of each cell and of the scale of each sector. Thus
an offset that is the same in all frames does not change the fit.
`RingMask.from_bands(setup, [(centre, half_width)], unit='q')` makes a mask
from ring bands that you know. The frame-window sizes of the method are for
0.2-degree frames. For another frame width the method scales them with
0.2 / width. This scaling is not validated.

Memory: pass A keeps 6 bytes per frame, sector and radial bin on the host
(about 1.3 GB for 1735 frames and 1785 radial bins). The table fit needs more
memory than the tables. Measured case (GPU, 1735 frames, 1785 radial bins,
grid 297 x 297 x 241, `keep_tables=True`, no `raw`, no halves): the peak
resident memory of the process was 6.0 GB above the memory before
`run_frame_chain`. The peak is in the table fit. Between the passes the
chain held 3.1 to 3.4 GB. The frames are streamed.
The GPU holds the model volume, the pixel vectors (12 bytes per pixel) and
the temporary arrays of one frame.
