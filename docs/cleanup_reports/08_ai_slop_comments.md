# Task 8 — AI slop / unnecessary comments

## Findings

Overall the codebase is already quite clean of typical AI slop. Most comments
encode real physics or reverse-engineered file-format information, and the
ones I do remove are narrow examples of restatement or structural demarcation.

Specific items flagged:

- `volume_builder.py:85-90` — Two consecutive banner comments (`Grid computation
  (from rsp_reader logic)` immediately followed by an empty banner and
  `Par file reading`). The first is stale (no function follows it before the
  second). Remove the dangling one.
- `volume_builder.py:293-298` — In-line comments on each `reshape/sum` line
  stating the resulting shape. These restate what the code does; the preceding
  "Sum one axis at a time..." comment already captures the rationale.
  Remove the shape annotations, keep the rationale.
- `volume_builder.py:692-698` — `_symmetrize_core` loop has trailing comments
  like `# (nh,nk,nl) float32 temp`, `# in-place accumulate`. These restate the
  operations. The preceding block-level comment already explains the memory
  strategy. Remove the trailing restatements.
- `volume_builder.py:399` — `# int32 storage — same as MATLAB's int32` — this
  is a borderline MATLAB-equivalence note, but `dtype=np.int32` on the next
  line already makes this clear. Keep — it's a correctness/consistency note.
- `volume_builder.py:417` — `# Also try par file` narrates the next block.
  Remove (the code is self-explanatory and adjacent code already describes
  par-file fallback).
- `volume_builder.py:1007` — `# Fallback: direct CELL line` inside
  `read_par_cell` — keep, this documents a branch that would otherwise be
  unclear from code (it's a secondary strategy).
- `rsp_viewer.py:77-79, 137-139, ...` — Multiple banner blocks using Unicode
  box-drawing characters. These are section demarcations that add ceremony.
  Most methods are well-named; removing them would not hurt readability.
  **Low-confidence — leaving in place**; they are consistent style in this
  file and a human reviewer may prefer to keep or strip them en masse.
- `volume_isosurface.py:248` — `# --- Isovalue slider (log scale) ---` and
  similar: these are section dividers. They separate small, clearly
  distinguishable blocks in a long function. **Low-confidence — leave**.
- `volume_builder.py:557` — `# axis_map: e.g. {'H': 'h', ...}` explains a
  parameter shape. Keep (aids understanding).
- `rsp_viewer.py:632` — `# CBF: pixel range` — keep, differentiates branch.
- `rsp_viewer.py:627` — `pass  # fallback: use unexpanded limits` — keep, it
  justifies a silent `pass`.
- `volume_builder_gui.py:390-394` — comments explaining the prefix-stripping
  example. Keep — they're a concrete example of the filename convention.
- `volume_builder.py:86` — `# ──── Grid computation (from rsp_reader logic) ────`
  appears to be a dead/orphan banner. There is no code between this banner
  and the next banner `# ──── Par file reading ────` at line 88. The actual
  grid computation code lives much later. Remove the orphan banner.

## Categorized removals

### Orphan/dangling banner comments
- `volume_builder.py` `Grid computation (from rsp_reader logic)` banner
  (lines ~85-87) — no associated code.

### Shape-annotation comments that restate code
- `volume_builder.py` in `bin_volume`: `# (nh_new, nk_t, nl_t)`,
  `# (nh_new, nk_new, nl_t)`, `# (nh_new, nk_new, nl_new)` trailing comments.
- `volume_builder.py` in `_symmetrize_core`: `# (nh,nk,nl) float32 temp`,
  `# broadcast bool, tiny`, `# (nh,nk,nl) bool temp`,
  `# in-place zero unmeasured`, `# in-place accumulate`,
  `# bool -> int16 in-place`, `# free immediately`.

### Narration comments
- `volume_builder.py:417` `# Also try par file`.

## Preservations (notable)

- All `.img` header offset comments (`# 278-281: NX, NY`, etc.) — domain-critical
  reverse-engineered format documentation.
- `volume_builder.py:680-684` — memory/precision rationale for
  `float32`+`int16` accumulators. Non-obvious numerical justification. **Keep**.
- `volume_builder.py:399` `int32 storage — same as MATLAB's int32` — MATLAB
  equivalence note. **Keep**.
- `volume_builder.py:1042-1046` — "KL: fixed H — per-row ih correction" and
  similar branch labels, and the `# HK M_inv for cross-term correction` note.
  These encode monoclinic-cell physics from CLAUDE.md. **Keep**.
- `volume_builder.py:85-87` grid note (NOT the banner; the prose comment in
  `compute_plane_M_inv`): explains "raw reciprocal vectors (not projected
  perpendicular to fixed axis)". **Keep** — CLAUDE.md calls this out as
  critical.
- `rsp_reader.py:186-190` — raw-vector explanation. **Keep**.
- `rsp_viewer.py:604-610, 246-248` — aspect-ratio math explanation and the
  reason for re-applying axis range on widget resize. **Keep**.
- `volume_builder.py` `_build_axis_permutation` inline comments —
  non-trivial permutation-building algorithm, comments aid reading. **Keep**.
- Header-offset comment in `_read_header_fast` (`f64(896)`, `f64(904)`,
  `f64(936)`, `f64(944)`): these correspond to the HK/HL/KL axis-flag offsets
  documented in CLAUDE.md. **Keep** (already minimal).
- `volume_builder.py:1098` `# Use the stored s directly — bin_volume already
  updates it correctly` — documents a non-obvious dependency (the bin_volume
  metadata-update fix from CLAUDE.md). **Keep**.
- `rsp_viewer.py:504-506` `# Add colorbar BEFORE applying axis range so the
  widget aspect reflects the final layout` — explains non-obvious ordering.
  **Keep**.

## Recommendations (low confidence — leaving for humans)

- The Unicode box-drawing banner comments (`# ──────...`) throughout
  `rsp_viewer.py`, `volume_builder.py`, and `volume_builder_gui.py` are
  stylistic; a human maintainer may want to keep or strip them wholesale.
- Docstrings are generally concrete. A couple could be tighter (e.g.
  `"""Bin a 1D array by averaging groups of b elements."""`) but they are
  not slop; leaving them alone.
- `volume_process.py:81` `# Prefix for output filenames` — borderline; it
  briefly labels a one-liner and aids scanning. Keep.

