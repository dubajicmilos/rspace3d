# Task 7 — Legacy / fallback / duplicate paths

## Duplicate logic found

### `_PLANE_CONFIG` dict literal (identical copies)
- `rspace3d/rsp_reader.py:51-58`
- `rspace3d/volume_builder.py:226-233`

Both files define the same dictionary mapping plane type to UB columns / axis
labels. The rsp_reader copy is private (`_PLANE_CONFIG`), as is the
volume_builder one. Unifying them requires one module to import the other.
Preference would be to keep the definition in `volume_builder.py` (it is the
lower-level module and `rsp_reader.py` already does not need volume_builder),
or in a shared `_common.py`. **Low confidence** — not changing in this pass
because deduplicating risks an import cycle and the duplication is a tiny
constant. Flagged for human review.

### Basis/M matrix construction (near-duplicate formulas)
- `rspace3d/rsp_reader.py:_compute_index_grid` (lines 168-218) — builds `M`,
  `M_inv` from raw reciprocal vectors, plus `idx1/idx2` meshgrids.
- `rspace3d/volume_builder.py:compute_plane_M_inv` (lines 206-223) — the same
  `M`/`M_inv` calculation without the grids.

The rsp_reader version includes additional outputs (grids, step sizes) that
`compute_plane_M_inv` doesn't need. Both use identical formulas. This is
technically a duplication but collapsing it would push grid computation
through an indirection. **Low confidence** — keep as is; one could call the
other, but the factoring is not cleaner than the current state.

### `_num` filename-sort helper (locally defined twice)
- `rspace3d/volume_builder_gui.py:272` (inside `_set_folder`)
- `rspace3d/volume_process.py:74` (inside `main`)

Both are 3-line local helpers with identical bodies. Moving to a module-level
utility is possible but adds an import. **Low confidence — leave**.

### Duplicated `base` prefix derivation
- `volume_builder_gui.py:278` and `volume_builder_gui.py:395` (inside
  `_do_process_all`) both derive the prefix from the first .img filename.
  The worker thread intentionally re-derives it so the worker doesn't depend
  on GUI state. **Intentional — keep**.

## Dead branches identified

None of the form `if False` / `if True` / unreachable `else`. The `else`
branch at `volume_builder.py:1074` looks suspicious at first glance (the
caller `extract_volume_slice` only dispatches to `_extract_nonnat` when the
view plane differs from `vol.plane_type`) but it IS reachable when viewing
the HK plane of a volume that has `plane_type='HL'` or `plane_type='KL'`.
Volumes with non-HK plane_type can be produced from unwarp folders where
the .img files are HL or KL reconstructions. **Keep**.

## Obsolete code

### Unused functions
- `rspace3d/rsp_reader.py:221 read_par_UB` — defined but never called from
  anywhere in the repo. `volume_builder.read_par_cell` supersedes it (reads
  UB + wavelength + cell from the .par file). **Remove.**
- `rspace3d/volume_builder.py:749 _to_hkl` and `:758 _from_hkl` — utility
  axis-remapping functions defined but never referenced. **Remove.**
- `rspace3d/volume_builder.py:1222 save_volume_npz` — defined but never
  called. The viewer still reads `.npz` files
  (`rsp_viewer.py:311-315`) but nothing produces them anymore. The writer
  is true dead code; the reader is kept for back-compat with old files.
  **Remove `save_volume_npz`, keep reader.**

### Unused instance attributes
- `rspace3d/volume_builder_gui.py:316-318` `self._file_count`,
  `self._frame_nx`, `self._frame_ny` — assigned in `_set_folder` and never
  read. **Remove.**
- `rspace3d/volume_builder_gui.py:278` `self._img_prefix` — only used one
  line later in a log message; does not need to be an instance attribute.
  **Convert to local.**

### Unused imports
- `rspace3d/rsp_viewer.py:19,22` — `QHBoxLayout`, `QStackedWidget`.
- `rspace3d/volume_builder_gui.py:23,28` — `QStatusBar`, `VolumeData`.
- `rspace3d/volume_isosurface.py:20` — `VolumeData` (only in docstring).
- `rspace3d/volume_process.py:24,25` — `re`, `numpy as np`.

All confirmed unused by AST scan and grep. **Remove all.**

## Intentionally preserved fallbacks

1. **CuPy GPU → NumPy CPU** — `volume_builder.py:594-601` (`_has_gpu`), and
   `symmetrize_volume` / `reject_outliers` `use_gpu` branches. Documented in
   CLAUDE.md.
2. **PyVista → Plotly** — `volume_isosurface.py:144-150` `except ImportError`
   falling back to `_plot_plotly`. Documented in CLAUDE.md.
3. **`.par` file primary, `.img` UB fallback** for cell parameters
   (`volume_builder.py:117 read_par_cell`, `volume_process.py:102-108`,
   `volume_builder_gui.py:292-299`). Both paths are real features.
4. **`.npz` reading** in `rsp_viewer.py:311-315` — kept to read legacy npz
   files users may still have (no new writer).
5. **CBF loader** in `rsp_viewer.py:295-304` — advertised feature
   (README.md, docstring).
6. **Plotly isosurface path** — used by CLI and notebook
   (`plot_isosurface_notebook`); the PyVista path is the interactive one
   used from the Qt viewer.
7. **Cell-line parsing fallback** in `read_par_cell`
   (`volume_builder.py:159-173`) — fallback when the UB+wavelength extraction
   path fails. Documented in the docstring.
8. **Silent fallback to `HK`** in `_read_header_fast` and `_detect_plane_type`
   — both default to HK when axis flags are ambiguous.

## Recommendations

### High confidence (applied)
1. Remove `read_par_UB` from `rsp_reader.py`.
2. Remove `_to_hkl`, `_from_hkl` from `volume_builder.py`.
3. Remove `save_volume_npz` from `volume_builder.py`.
4. Remove dead instance attributes `_file_count`, `_frame_nx`, `_frame_ny`
   from `volume_builder_gui.py:316-318`.
5. Demote `_img_prefix` to a local variable.
6. Remove unused imports (`QHBoxLayout`, `QStackedWidget`, `QStatusBar`,
   `VolumeData` x2, `re`, `numpy as np`).

### Low confidence (not applied — flagged for humans)
1. Unify duplicated `_PLANE_CONFIG` into one source. Requires either one
   module importing another or a tiny `_common.py`. Tradeoff between
   DRY and keeping `rsp_reader.py` self-contained.
2. Collapse `_compute_index_grid` and `compute_plane_M_inv` — possible but
   the two have different return shapes (grids vs 2x2) and the current
   separation mirrors their use sites.
3. The `_num` helper could move to a module-level utility, but it is a
   3-line helper used twice. Not worth the import.
4. `volume_process.py:86-87` reads the first-file header twice (line 84 and
   line 86) — inefficiency, not legacy. Flag only.
