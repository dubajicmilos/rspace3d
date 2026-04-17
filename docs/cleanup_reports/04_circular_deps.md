# Task 4 — Circular Dependency Analysis

## Method

`pydeps` installed but Graphviz unavailable on Windows host, so the graph was
built by AST walking every module for `ImportFrom` nodes with `level > 0`
or `module.startswith('rspace3d')`. Runtime check:

```
python -c "import rspace3d; from rspace3d import \
    rsp_reader, volume_builder, volume_isosurface, \
    make_dcunwarp, volume_process, volume_builder_gui, rsp_viewer"
```

passes cleanly, confirming no top-level cycles.

## Dependency graph

```
                      rsp_reader          make_dcunwarp
                         |                      |
                         v                      |
                    volume_builder              |
                         |                      |
         +---------------+-----------+          |
         v               v           v          |
  volume_isosurface  volume_process  |          |
         |                           |          |
         +---------+                 |          |
                   v                 v          v
               rsp_viewer       volume_builder_gui
```

Layering (strict, L0 = leaf):

| Layer | Module             | Depends on                                |
|-------|--------------------|-------------------------------------------|
| L0    | `rsp_reader`       | (none)                                    |
| L0    | `make_dcunwarp`    | (none)                                    |
| L1    | `volume_builder`   | `rsp_reader`                              |
| L2    | `volume_isosurface`| `volume_builder`                          |
| L3    | `volume_process`   | `volume_builder`                          |
| L3    | `volume_builder_gui`| `volume_builder`, `make_dcunwarp`         |
| L3    | `rsp_viewer`       | `rsp_reader`, `volume_builder`, `volume_isosurface` (deferred) |
| L3    | `__init__`         | re-exports from `rsp_reader`, `volume_builder`, `make_dcunwarp` |

## Cycles found

**None.** Every edge points strictly downward in the layer stack; no
A → B → A paths exist at module-load time or via deferred imports.

## Deferred imports found

| File:line                      | Import                                           | Status       |
|--------------------------------|--------------------------------------------------|--------------|
| `volume_builder.py:413` (was)  | `from .rsp_reader import read_rsp_layer as _read_layer` | **Fixed** — promoted to top-level (module already imported there for `_PLANE_CONFIG`; no cycle justified the laziness) |
| `volume_process.py:68` (was)   | `from rspace3d.volume_builder import _filter_numbered_imgs` | **Fixed** — promoted to top-level (same module already imported for 13 other names) |
| `rsp_viewer.py:846`            | `from .volume_isosurface import plot_isosurface` | **Left as-is** — pulls heavy optional deps (`plotly`, optionally `pyvista`) only when user clicks the isosurface button; legitimate lazy-load for an optional viewer |

## Fixes applied

1. `volume_builder.py` — moved `read_rsp_layer` into the existing top-level
   import (same line as `_PLANE_CONFIG`), removed the inline import in
   `load_unwarp_folder`.
2. `volume_process.py` — added `_filter_numbered_imgs` to the existing
   top-level import block, removed the inline import inside `main()`.

Both changes are equivalence transforms (module already loaded, same name
binding), verified by re-running the import check and `py_compile`
on all eight files.

## Coupling / cohesion observations (not fixed)

- **Private-name leakage.** `volume_builder_gui` and `volume_process` both
  import underscore-prefixed helpers from `volume_builder`
  (`_read_header_fast`, `_filter_numbered_imgs`, `_EXPECTED_ORDERS`).
  Since these two modules are the only callers, they are effectively
  "semi-public" — either rename them without underscore, or add an
  `__all__` and treat them as intentional package-internals. Low priority.
- **`rsp_viewer` breadth.** Depends on three siblings plus the isosurface
  module. Reasonable for a top-level GUI; no refactor needed.
- **`__init__` re-exports.** 15 names surfaced; matches the package's
  documented public API. No issues.

## Left as-is

- The `rsp_viewer` → `volume_isosurface` deferred import — keeps GUI
  startup fast and avoids forcing pyvista/plotly on users who never open
  the 3D view. Correct design.
- Private-name imports — flagged above but changing them touches the
  public surface; outside scope of "untangle circular deps."
