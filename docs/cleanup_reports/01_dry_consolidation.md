# Task 1 — DRY Consolidation

Scanned all 8 files in `rspace3d/` for duplication. Prior agents consolidated
`_PLANE_CONFIG` (Task 2) and flagged several candidates as "low confidence"
(Task 7). This task validates and implements the high-confidence ones.

## Duplications found

### 1. `_num` filename-sort helper (identical, 3 lines x 2 sites)
- `volume_process.py:74-76` (inside `main`)
- `volume_builder_gui.py:275-277` (inside `_set_folder`)

Identical body:
```python
def _num(f: str) -> int:
    try: return int(f.rsplit('_', 1)[1].split('.')[0])
    except (ValueError, IndexError): return 0
```
Both are immediately followed by `sorted(img_files, key=_num)`. Tight coupling
to `_filter_numbered_imgs`, which already lives in `volume_builder`.

### 2. M-matrix formula (identical math, different return shapes)
- `rsp_reader._compute_index_grid` (lines 171-237) builds the 2x2 M from raw
  reciprocal vectors, then builds idx1/idx2 meshgrids, s, cx, cy, step sizes.
- `volume_builder.compute_plane_M_inv` (lines 212-231) does only the 2x2 step.

Validated: identical (max diff 0.0 across all three planes on the monoclinic
test cell). CLAUDE.md "NO PROJECTION" invariant holds in both.

### 3. "Load cell with par-file-then-UB fallback" (6-line block x 2 sites)
- `volume_process.py:102-108`
- `volume_builder_gui.py:294-301`

Both do: `find_par_file(folder) -> read_par_cell(par) -> fall back to
cell_from_ub(hdr['ub'], hdr['wavelength'])`. The GUI variant logs along the
way; the CLI variant does not. A helper that returns `(cell, source_label)`
preserves both logging paths without branching.

A related-but-different variant exists in `load_unwarp_folder`
(`volume_builder.py:414-420`). It starts from the UB-header cell and
OVERRIDES with par-file cell on success — opposite preference to the GUI/CLI
blocks above. Intentional, NOT a duplicate.

### 4. Plotly isosurface figure construction
- `_plot_plotly` (lines 113-164)
- `plot_isosurface_notebook` (lines 386-434)

Both build a `go.Figure`, add `go.Isosurface` traces from a meshgrid of H/K/L,
set the same scene labels ("h","k","l"), aspectmode="data". The only real
differences: notebook returns the figure instead of calling `.show()`, and
the notebook skips `save_html`. One helper that builds the Figure, two
shallow wrappers.

## Consolidations implemented

| Canonical location | What | Replaces |
|---|---|---|
| `volume_builder._img_number(fname)` | 3-line filename-suffix int parser | two local `_num` helpers |
| `rsp_reader.compute_plane_M_inv` (moved) + re-export from `volume_builder` | pure M_inv-from-UB | inlined copy in `rsp_reader._compute_index_grid`; delegates |
| `volume_builder.resolve_unit_cell(folder, header, log=None)` | par-file-first cell resolution | two 6-line copy-pasted blocks |
| `volume_isosurface._build_plotly_figure(...)` | shared Plotly figure builder | duplicated trace/layout block |

All four are pure-function helpers with clear single responsibilities. No new
module created; helpers live next to their primary use sites.

## Kept separate (with rationale)

- **`_PLANE_CONFIG`** — already consolidated by Task 2.
- **`load_unwarp_folder` cell-resolution block** — intentionally *opposite*
  preference from the GUI/CLI blocks (UB first, par override). Merging would
  force an extra flag. Kept as-is.
- **Non-identical "resolve par path + log" variations** — the GUI logs
  `f'Par file: {os.path.basename(par_path)}'` before reading; the CLI logs
  `f'  Par:     {os.path.basename(par_path)}'` (different spacing). The
  `resolve_unit_cell` helper accepts a `log` callback so both preserve
  formatting.
- **`_extract_native` vs `_extract_nonnat` / `extract_volume_slice`** —
  already a single source of truth (Task 6 / CLAUDE.md). Not touched.
- **`_read_header_fast` vs `_detect_plane_type` in `rsp_reader`** — the
  former is a minimal fast header parser that skips intensity read; the
  latter is the detection branch inside the full reader. Different roles.

## Implementation notes

1. **`compute_plane_M_inv` moved to `rsp_reader`**: it operates purely on
   UB matrix and `_PLANE_CONFIG` (which already lives in `rsp_reader`), so
   moving it eliminates the only cross-module reference in the formula.
   `volume_builder` re-exports it so downstream imports (`rsp_viewer.py`,
   public API via `__init__.py`, `volume_builder._extract_nonnat`) don't
   break. `_compute_index_grid` now calls `compute_plane_M_inv` instead of
   re-deriving.

2. **`_img_number` placement**: lives in `volume_builder` next to
   `_filter_numbered_imgs` (the other filename-parsing helper). Both
   `volume_process.py` and `volume_builder_gui.py` already import from
   `volume_builder`, so no new imports needed.

3. **`resolve_unit_cell(folder, header, log=None)`**: takes an optional
   logger callback. Returns `dict[str, float]`. Preserves each call-site's
   log format.

4. **`_build_plotly_figure`**: shared Plotly builder. `_plot_plotly` wraps
   it with `save_html` + `.show()`; `plot_isosurface_notebook` just returns
   the figure.

## Verification

```
$ python -m py_compile rspace3d/*.py                      OK
$ python -c "import rspace3d; from rspace3d import ..."   OK (all 8 modules)
$ python -m mypy --ignore-missing-imports rspace3d/       Success: no issues
```

Regression test on `example_data/example_monoclinic.h5`:
- volume loads identically (419 x 368 x 321, plane=HK).
- M_inv for all 3 planes matches pre-change to 0.0 diff.
- `extract_volume_slice` sums for HK / HL / KL match pre-change to 0.0 diff.

## Summary

- 4 duplications consolidated, ~25 net lines removed.
- 0 new modules.
- Public API unchanged (`compute_plane_M_inv` still importable from
  `volume_builder` and `rspace3d`).
- No mypy regression.
