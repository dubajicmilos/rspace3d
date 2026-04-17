# Task 6 — Defensive programming (try/except, hasattr, getattr)

## Search commands

```
grep -n "try:" rspace3d/*.py            # 14 matches
grep -n "except" rspace3d/*.py          # 13 matches
grep -n "hasattr(" rspace3d/*.py        # 0 matches
grep -n "getattr(" rspace3d/*.py        # 0 matches
grep -n "is not None else" rspace3d/*.py  # 0 matches
grep -n "except:" rspace3d/*.py         # 2 bare excepts
```

## Catalogue

| # | File : line | Pattern | Verdict | Rationale |
|---|-------------|---------|---------|-----------|
| 1 | `volume_builder.py:142-148` | `except (ValueError, IndexError): pass` around UB-matrix parsing in `read_par_cell` | **KEEP** | Parsing external `.par` file format. Specific exceptions. Fallback = try next strategy (CELL line). Documented in docstring. |
| 2 | `volume_builder.py:152-156` | `except (ValueError, IndexError): pass` around wavelength parsing | **KEEP** | Same as above — external-file parsing with specific exceptions. |
| 3 | `volume_builder.py:161-173` | `except (ValueError, IndexError): pass` around CELL-line parsing | **KEEP** | Same — the documented fallback path for cell parameters. |
| 4 | `volume_builder.py:593-598` | `try: import cupy ... except Exception: return False` in `_has_gpu` | **KEEP** | Documented CuPy/CPU fallback (CLAUDE.md). Exception type broad because CUDA drivers can fail in many ways at runtime. |
| 5 | `volume_isosurface.py:143-149` | `try: import pyvista ... except ImportError: fall back to plotly` | **KEEP** | Documented PyVista/Plotly fallback (CLAUDE.md). |
| 6 | `volume_isosurface.py:225-232` | `try: ... except Exception: pass` around VTK `contour()` in slider-rebuild callback | **REPLACE** | Callback runs on every slider move. Bare-swallow hides real bugs. Narrow to specific exceptions (`RuntimeError`, `ValueError`) and print to `stderr` so the operator sees problems without the GUI dying. |
| 7 | `rsp_viewer.py:273-282` | `try: load file ... except Exception: status + traceback` in `_load_file` | **KEEP** | User-supplied file at system boundary (img/cbf/h5/npz). Shows error in status bar and prints traceback — not silent. |
| 8 | `rsp_viewer.py:611-627` | `try: compute widget aspect ... except Exception: pass # fallback: use unexpanded limits` | **KEEP** | Documented fallback (comment says so). matplotlib widget extent is not always available during resize events. |
| 9 | `rsp_viewer.py:840-848` | `try: launch isosurface viewer ... except Exception: status message` | **KEEP** | Launching an optional 3D viewer — the main viewer must survive a failure there. Status bar shows the error. |
| 10 | `volume_process.py:72-74` | `try: return int(...) except: return 0` in `_num` (filename sort key) | **REPLACE** | **Bare `except:` catches KeyboardInterrupt/SystemExit.** Files are pre-filtered by regex `_\d+\.img$` so format is known. Narrow to `(ValueError, IndexError)` defensively but remove the bare-except smell. |
| 11 | `volume_builder_gui.py:59-64` | `try: run func ... except Exception: emit error signal` in `WorkerThread.run` | **KEEP** | Qt `QThread.run` override. Must not let exceptions escape the thread — they would be lost. Traceback is emitted via signal, not swallowed. |
| 12 | `volume_builder_gui.py:82-87` | `try: read GPU name ... except Exception: log "GPU detected"` | **KEEP** | Inner fallback when CuPy device-name lookup fails but GPU is already detected (outer `if HAS_GPU`). Degrades gracefully in log only. |
| 13 | `volume_builder_gui.py:273-274` | `try: return int(...) except: return 0` — same `_num` helper | **REPLACE** | Same as #10 — bare `except:` smell. Narrow to specific exceptions. |
| 14 | `volume_builder_gui.py:339-343` | `try: chdir ... finally: restore cwd` | **KEEP** | `try/finally`, not `try/except`. Essential cleanup pattern. |

No `hasattr(...)`, `getattr(..., default)`, or `x if x is not None else y` defensive reads found.

## Changes applied

- `volume_process.py:72-74` — bare `except:` narrowed to `(ValueError, IndexError)`.
- `volume_builder_gui.py:273-274` — bare `except:` narrowed to `(ValueError, IndexError)`.
- `volume_isosurface.py:225-232` — `except Exception: pass` narrowed to
  `(RuntimeError, ValueError)` and now prints a one-line warning to stderr so
  the operator can see when an isovalue/clip combination fails to render.

## Verification

```
python -c "import rspace3d; from rspace3d import rsp_reader, volume_builder, volume_isosurface, make_dcunwarp, volume_process, volume_builder_gui, rsp_viewer"
python -m py_compile rspace3d/*.py
```

Both commands succeed with no output.

## Summary counts

- **Found**: 13 try/except (+ 1 try/finally, not counted).
- **Kept**: 10 (file-parsing, Qt/VTK boundaries, documented fallbacks, QThread, try/finally).
- **Replaced with narrower exceptions**: 3 (two filename `_num` helpers with bare `except:`, one VTK callback).
- **Removed entirely**: 0.

## Flagged for later review

- `volume_isosurface.py` contour-callback: the narrowed exception set
  (`RuntimeError, ValueError`) is a best guess for VTK/pyvista failures on
  pathological isovalues. If a different exception type ever leaks through,
  the operator will now see a cryptic stack trace instead of a silent no-op.
  Acceptable tradeoff — the previous `except Exception: pass` was hiding
  everything, including real bugs.
