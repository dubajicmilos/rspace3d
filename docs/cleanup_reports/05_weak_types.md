# Task 5 — Strengthen Weak Type Annotations

## Baseline

```
$ python -m mypy rspace3d/ --ignore-missing-imports
rspace3d\volume_builder.py:163: Incompatible types in assignment
  (expression has type "list[str]", variable has type "list[float]")
Found 1 error in 1 file (checked 8 source files)
```

`pyproject.toml` sets `requires-python = ">=3.10"`, so PEP 604 and builtin
generics are available.

## Catalogued weaknesses (all high confidence)

| File | What | Proposed |
|------|------|----------|
| `rsp_reader.py` | `RSPLayer` bare `np.ndarray` / `dict` fields | `NDArray[np.int32/np.float64]`, `tuple[float, float]` |
| `rsp_reader.py` | `_PLANE_CONFIG` untyped (mixed int/str values) | `dict[str, dict[str, Any]]` |
| `rsp_reader.py` | `read_rsp_layer` / `_compute_index_grid` unsigned | typed signatures |
| `volume_builder.py` | `from typing import List, Tuple, Optional, Dict` | PEP 604 / builtins |
| `volume_builder.py` | `VolumeData` bare `np.ndarray` / `dict` | `NDArray[Any/np.floating]`, `dict[str, Any]` |
| `volume_builder.py:163` | `vals` reused with incompatible types | rename `ub_vals` / `cell_strs` (fixes baseline) |
| `volume_builder.py` | `progress_callback=None` untyped | `Callable[[int, int], None] \| None` |
| `volume_builder.py` | `_LAUE_GROUPS_CACHE = {}` bare | `dict[str, list[NDArray[np.int_]]]` |
| `volume_builder.py` | `dim_to_std = [None, None, None]` then assigned ints | `list[int] = [0, 0, 0]` |
| `volume_builder.py` | `extract_volume_slice` / `_extract_*` | full typed signatures |
| `make_dcunwarp.py` | `generate_dcunwarp(...)` untyped | `-> list[str]` with float params |
| `volume_isosurface.py` | `plot_isosurface(vol, isovalue=None, ...)` untyped | `VolumeData \| str`, `float \| list[float] \| None` |
| `volume_builder_gui.py` | `WorkerThread.__init__(self, func, *args)` untyped | `Callable[..., Any]`, `Any` |
| `volume_builder_gui.py` | `self._worker`/`_folder_path = None` bare | `WorkerThread \| None`, `str \| None` |
| `rsp_viewer.py` | state attrs (`self.layer`, `self.vol`, ...) untyped | `RSPLayer \| None`, `VolumeData \| None`, ... |

## Notes

- Added `from __future__ import annotations` to all edited modules.
- `VolumeData.H/K/L` uses `NDArray[np.floating]` (not `np.float64`) — `bin_1d`
  returns `floating[Any]`; tighter typing would force visible casts.
- `_PLANE_CONFIG` values are genuinely heterogeneous (int columns + str labels).
- `metadata` stays `dict[str, Any]`: keys change through the pipeline (cell
  params, UB, lambda, laue_group, s, cx, cy, M_inv, n_outliers_replaced, ...).
  A TypedDict would lock one stage and fight `.get()` calls downstream.
- Three narrow `# type: ignore` remain for legitimate stub-over-strict cases:
  `QMainWindow.statusBar()` and `QThread.currentThread()` (both stubbed
  `| None`), and PyVista `add_mesh(cmap=str)` / `set_background(...)` /
  `add_axes(...)` where stubs demand `Literal[...]` or reject documented kwargs.

## Verification

```
$ python -m mypy rspace3d/ --ignore-missing-imports
Success: no issues found in 8 source files

$ python -m py_compile rspace3d/*.py     (no output)
$ python -c "import rspace3d; from rspace3d import ..."   imports OK
```

## Summary

- Mypy errors: **1 → 0**.
- Files modified: **7**.
- Public API shape unchanged — only annotations added/tightened.
