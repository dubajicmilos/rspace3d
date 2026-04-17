# Task 2 — Shared Type Consolidation

## Types enumerated

### Dataclasses (2)
- `RSPLayer` (`rsp_reader.py:27`) — 2D reciprocal-space slice: `intensity`,
  `idx1`, `idx2`, `fixed_value`, `plane_type`, `x_label`, `y_label`,
  `fixed_label`, `thickness`, `M_inv`, `s`, `cx`, `cy`, `step_idx1`, `step_idx2`.
- `VolumeData` (`volume_builder.py:24`) — 3D volume: `intensity`, `H`, `K`,
  `L`, `plane_type`, `metadata`.

### Qt subclasses (not data types)
- `WorkerThread(QThread)` (`volume_builder_gui.py:44`)
- `SimpleVolumeGUI(QMainWindow)` (`volume_builder_gui.py:74`)
- `UnifiedViewer(QMainWindow)` (`rsp_viewer.py:47`)

### Constant dicts (module-level)
- `_PLANE_CONFIG` in `rsp_reader.py:53` — keys `{vec1_col, vec2_col, x_label,
  y_label, fixed_label}`.
- `_PLANE_CONFIG` in `volume_builder.py:232` — keys `{vec1_col, vec2_col}`.
- `_FIXED_OFFSETS`, `_THICKNESS_MIN`, `_THICKNESS_MAX` (`rsp_reader.py:48-50`)
  — header byte offsets, used only in rsp_reader.
- `_INV`, `_C2a`, …, `_C6_hex` matrices + `_EXPECTED_ORDERS`,
  `_LAUE_GENERATORS`, `_LAUE_GROUPS_CACHE`, `LAUE_GROUP_NAMES`
  (`volume_builder.py:479-535`). `LAUE_GROUP_NAMES` and `_EXPECTED_ORDERS` are
  re-imported by `volume_process.py` and `volume_builder_gui.py`.
- `COLORMAPS` list in `rsp_viewer.py:40` (UI only).
- `_get_axis_mapping` returns an inline 3-plane dict (`volume_builder.py:542`).

### No TypedDicts / Enums / NamedTuples / Protocols / TypeAliases exist in
the package. All module-level type annotations use the builtins plus
`npt.NDArray[...]`.

## Duplicates found

### `_PLANE_CONFIG` — genuine duplication (overlapping, not identical)
Both files define a dict keyed by `'HK'|'HL'|'KL'`. The `rsp_reader` version
is a **superset** (5 fields) of the `volume_builder` version (2 fields). Every
reader of the volume_builder copy accesses only `vec1_col` / `vec2_col`, which
are also present (identical values) in the rsp_reader copy. The Task 7 report
flagged this as "low confidence" due to import-cycle risk.

**No cycle exists.** `volume_builder` already imports from `rsp_reader`
(`from .rsp_reader import read_rsp_layer as _read_layer`); `rsp_reader` does
not import from `volume_builder`. Merging is safe.

## Consolidation plan (applied)

1. **Remove** the local `_PLANE_CONFIG` in `volume_builder.py:232-236`.
2. **Add** `from .rsp_reader import _PLANE_CONFIG` at the top of
   `volume_builder.py`.
3. The `rsp_reader.py` definition stays put (it is the superset, and keeping
   it in the lower-level module with the detection code where the labels are
   also needed is the natural home).

Rationale: single source of truth; no cycle; no API surface change
(`_PLANE_CONFIG` is private — `_` prefix — to the package). The
subset-vs-superset distinction was cosmetic; the unused keys cost nothing.

## Kept separate (with rationale)

- **`RSPLayer` vs `VolumeData`**: intentionally different shapes (2D slice vs
  3D volume) with different purposes. Merging would require `Optional[...]`
  fields that make both APIs worse.
- **Plane-name literals `"HK" | "HL" | "KL"`**: a `Plane = Literal["HK",
  "HL", "KL"]` alias would touch 15+ function signatures across 4 files for
  marginal benefit. `plane_type` strings are validated once at
  `_detect_plane_type` and constructed only from that source. Not worth the
  churn.
- **`_FIXED_OFFSETS` / `_THICKNESS_MIN` / `_THICKNESS_MAX`**: used only in
  `rsp_reader.py`. Not shared.
- **`LAUE_GROUP_NAMES` / `_EXPECTED_ORDERS`**: already shared via re-import
  from `volume_builder`. Adequate.
- **No new `_types.py` module**: would contain one dict. Over-engineering
  for a 7-file package.

## Verification

```
$ python -m py_compile rspace3d/*.py             → OK
$ python -c "import rspace3d; from rspace3d import ..."  → all imports OK
$ python -m mypy --ignore-missing-imports rspace3d/
  Success: no issues found in 8 source files
```

Baseline (before): 0 mypy errors.  After: 0 mypy errors. No regression.

## Summary

- Duplicates eliminated: **1** (`_PLANE_CONFIG`).
- New shared modules created: **0**.
- Public API change: **none**.
- Files modified: **1** (`volume_builder.py`: removed 5-line dict, added
  1-line import).
