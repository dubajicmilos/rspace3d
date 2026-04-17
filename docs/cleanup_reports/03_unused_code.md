# Task 3: Unused Code Detection & Removal

## Tools used + output

- `vulture rspace3d/ --min-confidence 60` (via `python -m vulture`)
- `pyflakes rspace3d/` (via `python -m pyflakes`)
- Manual `Grep` across the whole tree (`rspace3d/`, `scripts/`,
  `tests/`, `notebooks/*.ipynb`, `pyproject.toml`, `*.spec`, `docs/`).
  The notebook `notebooks/volume_analysis.ipynb` imports
  `plot_isosurface_notebook`, `load_volume_h5`, `compute_plane_M_inv`,
  and `read_rsp_layer`; those are NOT dead.

### Initial vulture output (min-confidence 60)

```
rsp_reader.py:169        unused variable 'fixed_col'     (100%)
rsp_viewer.py:502        unused attribute 'format_coord'  (60%) — matplotlib API
rsp_viewer.py:733        unused attribute 'mode'          (60%) — matplotlib API
volume_builder.py:753    unused function 'compute_outlier_stats' (60%)
volume_builder_gui.py:58 unused method 'run'              (60%) — Qt QThread override
volume_isosurface.py:338 unused attribute 'camera_position' (60%) — pyvista API
volume_isosurface.py:342 unused function 'plot_isosurface_notebook' (60%) — used in notebook
```

### Initial pyflakes output (significant items only)

```
rsp_viewer.py:855        local 'import numpy as np' shadowing module-level np
volume_builder.py:994    local variable 'L' assigned but never used
volume_builder.py:1010   local variable 'cx' assigned but never used
volume_isosurface.py:58  local variable 'cell' assigned but never used
__init__.py:13–28        re-exports flagged as unused — public API, not dead
```

## High-confidence unused (removed)

| File | Lines (before) | What | Why safe |
|------|---------------|------|----------|
| `rspace3d/rsp_reader.py` | 47–48 | Constants `_XAXIS_FLAGS`, `_YAXIS_FLAGS` | Defined but never referenced. The axis-detection code on lines 148–151 hard-codes the offsets (896/904/936/944) directly. |
| `rspace3d/rsp_reader.py` | 169 | Parameter `fixed_col` of `_compute_index_grid` | Passed in by caller but never read inside the function body. Removed parameter and removed corresponding `'fixed_col'` keys from `_PLANE_CONFIG` (the only caller). |
| `rspace3d/rsp_reader.py` | 52–58 | `_PLANE_CONFIG` dict: `'fixed_col'` keys | Only `vec1_col`, `vec2_col`, `x_label`, `y_label`, `fixed_label` are read elsewhere. |
| `rspace3d/volume_builder.py` | 227–232 | `_PLANE_CONFIG` dict: `'fixed_col'`, `'x_label'`, `'y_label'`, `'fixed_label'` keys | Only `vec1_col` and `vec2_col` are used (inside `compute_plane_M_inv`). The `extract_volume_slice` function uses a different, inline tuple (`cfgs`) at line 948 for labels. |
| `rspace3d/volume_builder.py` | 753–779 | Function `compute_outlier_stats` | Not called anywhere in `rspace3d/`, tests, scripts, or the notebook. Was callable from the removed `volume_gui.py` (Task 7 prior). |
| `rspace3d/volume_builder.py` | 994 | Local `L` in `_extract_nonnat` (from `H, K, L = vol.H, vol.K, vol.L`) | `L` never read. Changed unpack to `H, K = vol.H, vol.K`. |
| `rspace3d/volume_builder.py` | 1010 | Local `cx = vol.metadata.get('cx', ...)` in `_extract_nonnat` | Never read. (Separate `cx_out` variable on line 1072 is fine.) |
| `rspace3d/volume_isosurface.py` | 58 | Local `cell = vol.metadata.get('cell')` in `plot_isosurface` | Assigned but never referenced. |
| `rspace3d/rsp_viewer.py` | 855 | `import numpy as np` inside `_show_info` | Module-level `import numpy as np` already exists (line 15); local import is redundant and `np` is not used inside the method anyway. |

## Flagged but preserving (with reason)

- **`run()` in `WorkerThread`** (`volume_builder_gui.py:58`) — Qt `QThread` override.
  Called by Qt's thread dispatcher when `.start()` is invoked.
- **`format_coord = lambda x, y: ''`** (`rsp_viewer.py:502`) — matplotlib
  `Axes.format_coord` API for suppressing the default cursor readout.
- **`self.nav_toolbar.mode = ''`** (`rsp_viewer.py:733`) — matplotlib
  `NavigationToolbar2QT.mode` attribute, set to clear the toolbar's current mode.
- **`pl.camera_position = 'iso'`** (`volume_isosurface.py:338`) — PyVista
  `Plotter.camera_position` API.
- **`plot_isosurface_notebook`** (`volume_isosurface.py:342`) — imported and
  called by `notebooks/volume_analysis.ipynb` section 5 (not detectable by
  static analysis against Python sources alone).
- **`__init__.py` re-exports** (lines 13–28) — explicit public API; flagged
  because there's no `__all__`. Left alone (adding `__all__` is a separate
  concern).
- **All `_on_*` / `_build_ui` / `_load_*` / `_display_*` / `_toggle_*` / `_lp_*`
  methods** in `rsp_viewer.py` and `volume_builder_gui.py` — each has exactly
  one caller (a Qt signal connection or single `__init__` dispatch), which is
  normal for GUI callbacks, not dead code.
- **All Laue-group generator matrices** (`volume_builder.py:_C2a`, `_C2b`, etc.)
  — each appears in `_LAUE_GENERATORS`; any change would silently break symmetry.
- **F-string "missing placeholders"** warnings from pyflakes (e.g.
  `lines.append(f'')`) — stylistic, not dead code; out of scope.

## Verification

```
$ python -c "import rspace3d; from rspace3d import rsp_reader, volume_builder,
      volume_isosurface, make_dcunwarp, volume_process, volume_builder_gui,
      rsp_viewer"
(no output — success)

$ python -m py_compile rspace3d/*.py
(no output — success)

$ python -m vulture rspace3d/ --min-confidence 60
(only the 5 false positives listed above remain)
```

## Summary

- **5 files modified**: `rsp_reader.py`, `volume_builder.py`,
  `volume_isosurface.py`, `rsp_viewer.py`, and the shared `_PLANE_CONFIG`
  in `volume_builder.py`.
- **Removed**: 2 unused module-level constants, 1 unused function (30 LOC),
  3 unused local variables, 1 redundant import, 1 unused parameter, and
  5 unused dict keys in `_PLANE_CONFIG` (across 2 modules).
- **Nothing user-visible changed**: no public symbol removed, no API signature
  changed except the private `_compute_index_grid(..., fixed_col)` helper.
- **Imports and compilation verified clean.**
