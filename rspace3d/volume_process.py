"""
volume_process.py — Command-line 3D volume processing.

One-line usage:
    python volume_process.py path/to/unwarp
    python volume_process.py path/to/unwarp --laue m-3m --sigma 5 --bin 4

Full pipeline: load (with coverage mask) -> save raw -> bin
               -> single-pass symmetrize + outlier rejection -> save.

Options:
    --laue    Laue group (default: m-3m)
              Choices: -1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm, m-3, m-3m
    --sigma   Outlier rejection sigma (default: 3.0); pass 0 to skip rejection
    --bin     HK bin factor (default: 2)
    --binl    L bin factor (default: 1)
    --no-gpu  Force CPU even if GPU available
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import numpy as np

from rspace3d.volume_builder import (
    load_unwarp_folder, bin_native, symmetrize_volume,
    save_volume_h5, load_volume_h5, resolve_unit_cell, raw_cache_mismatch,
    _read_header_fast, _filter_numbered_imgs, _img_number,
    LAUE_GROUP_NAMES, _EXPECTED_ORDERS, HAS_GPU,
)


def main() -> None:
    """Console entry point: a data error is reported on one line, not as a traceback."""
    try:
        _main()
    except (ValueError, FileNotFoundError, RuntimeError) as e:
        print(f'\nError: {e}', file=sys.stderr)
        sys.exit(1)


def _main() -> None:
    parser = argparse.ArgumentParser(
        description='Process CrysAlisPro unwarp .img files into symmetrized 3D volume.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python volume_process.py path/to/unwarp\n'
               '  python volume_process.py path/to/unwarp --laue 4/mmm --sigma 1.5 --bin 4\n'
               '  python volume_process.py path/to/unwarp --sigma 0   # no rejection\n'
               '  python volume_process.py path/to/unwarp --no-gpu\n')

    parser.add_argument('folder', help='Path to unwarp folder containing .img files')
    parser.add_argument('--laue', default='m-3m',
                        choices=list(LAUE_GROUP_NAMES.keys()),
                        help='Laue group (default: m-3m)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Outlier rejection sigma (default: 3.0); '
                             'pass 0 to skip rejection (pure symmetrization)')
    parser.add_argument('--morph', default='auto',
                        help="Coverage-mask kernel side (default: 'auto'). "
                             "'auto' picks ~5%% of one Bragg cell from the "
                             "header geometry; pass an int (3, 5, 7, ...) "
                             "to force a fixed pixel kernel.")
    parser.add_argument('--bin', type=int, default=2,
                        help='HK bin factor (default: 2)')
    parser.add_argument('--binl', type=int, default=1,
                        help='L bin factor (default: 1)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU even if GPU available')
    parser.add_argument('--force-reload', action='store_true',
                        help='Re-read .img files and overwrite existing '
                             '_raw.h5. Default: reuse the cached _raw.h5 '
                             'if its morph_size matches the request.')
    parser.add_argument('--workers', type=int, default=0,
                        help='Parallel file-load threads (0 = auto: min(8, cpu_count); '
                             '1 = sequential)')

    args = parser.parse_args()
    folder = args.folder
    use_gpu = HAS_GPU and not args.no_gpu

    if not os.path.isdir(folder):
        print(f'Error: {folder} is not a directory')
        sys.exit(1)

    # Count files (filter to single prefix)
    img_files = _filter_numbered_imgs(folder)
    if not img_files:
        print(f'Error: no numbered .img files found in {folder}')
        sys.exit(1)

    sorted_f = sorted(img_files, key=_img_number)
    n = len(sorted_f)

    # Prefix for output filenames
    base = sorted_f[0].rsplit('_', 1)[0]

    # Header info
    hdr = _read_header_fast(os.path.join(folder, sorted_f[0]))
    nx, ny = hdr['nx'], hdr['ny']
    l_min = _read_header_fast(os.path.join(folder, sorted_f[0]))['fixed_value']
    l_max = _read_header_fast(os.path.join(folder, sorted_f[-1]))['fixed_value']
    l_step = (l_max - l_min) / max(n - 1, 1)

    laue_clean = args.laue.replace('/', '').replace('-', 'bar')
    device = 'GPU' if use_gpu else 'CPU'

    print('=' * 60)
    print(f'  3D Volume Processor ({device})')
    print('=' * 60)
    print(f'  Folder:  {folder}')
    print(f'  Files:   {n}  |  {nx} x {ny} px  |  {hdr["plane_type"]}')
    print(f'  Lambda:  {hdr["wavelength"]:.5f} A')
    print(f'  Fixed:   {l_min:.3f} to {l_max:.3f} (step {l_step:.4f})')

    # Unit cell from par file (primary) or .img header (fallback)
    cell, par_path = resolve_unit_cell(folder, hdr)
    if par_path:
        print(f'  Par:     {os.path.basename(par_path)}')
    print(f'  Cell:    a={cell["a"]:.5f}  b={cell["b"]:.5f}  c={cell["c"]:.5f} A')
    print(f'           alpha={cell["alpha"]:.3f}  beta={cell["beta"]:.3f}  '
          f'gamma={cell["gamma"]:.3f} deg')

    sigma_val: float | None = args.sigma if args.sigma > 0 else None
    morph_arg: int | str = 'auto' if str(args.morph).lower() == 'auto' else int(args.morph)
    print(f'  Laue:    {args.laue} ({_EXPECTED_ORDERS[args.laue]} ops)')
    print(f'  Sigma:   '
          f'{sigma_val if sigma_val is not None else "off (pure average)"}')
    print(f'  Morph:   {morph_arg}')
    print(f'  Binning: {args.bin}x{args.bin} (HK), {args.binl}x (L)')
    print('=' * 60)

    raw_path = os.path.join(folder, f'{base}_raw.h5')

    # Compute what morph_size WOULD resolve to with current request — used
    # for cache validation. For 'auto' we apply the same formula the loader
    # would apply, using the header we already have.
    if morph_arg == 'auto':
        from rspace3d.volume_builder import adaptive_morph_size as _amorph
        s_pp_now = 2.0 / (hdr['d_min'] * hdr['nx'])
        plane_to_cols = {'HK': (0, 1), 'HL': (0, 2), 'KL': (1, 2)}
        c1, c2 = plane_to_cols[hdr['plane_type']]
        ub = np.asarray(hdr['ub']).reshape(3, 3)
        wl = hdr['wavelength']
        recip_now = 0.5 * (float(np.linalg.norm(ub[:, c1] / wl))
                           + float(np.linalg.norm(ub[:, c2] / wl)))
        morph_now: int = _amorph(s_pp_now, recip_now)
    else:
        morph_now = int(morph_arg)

    # Cache decision: the raw cache must carry the requested mask kernel AND
    # the fingerprint of the current .img/.par files.
    use_cache = False
    if os.path.isfile(raw_path) and not args.force_reload:
        try:
            cached = load_volume_h5(raw_path)
            reason = raw_cache_mismatch(cached, folder, morph_now)
            if reason is None:
                use_cache = True
            else:
                print(f'\n  Cache invalid: {os.path.basename(raw_path)}: {reason}. '
                      'Regenerating from .img files.')
        except Exception as e:
            print(f'\n  Could not read {os.path.basename(raw_path)} '
                  f'({e}); regenerating.')

    # ── Step 1+2: Load (cache or fresh) ──
    if use_cache:
        print(f'\n[1+2/4] Loading cached {os.path.basename(raw_path)}...',
              end=' ', flush=True)
        t0 = time.time()
        vol = cached
        dt = time.time() - t0
        nh, nk, nl = vol.intensity.shape
        print(f'{dt:.1f}s  ({nh}x{nk}x{nl}, {vol.intensity.dtype})')
        ms = vol.metadata.get('morph_size', morph_now)
        mode = vol.metadata.get('morph_mode', 'unknown')
        nu_mean = vol.metadata.get('n_unmeasured_per_frame_mean')
        nu_pct = vol.metadata.get('n_unmeasured_per_frame_pct')
        print(f'        Cached mask: morph={ms}px ({mode})'
              + (f', unmeasured/frame mean={nu_mean:,.0f} '
                 f'({nu_pct:.2f}%)' if nu_mean is not None else ''))
        print(f'        (skipped .img read + raw save; pass --force-reload '
              f'to regenerate)')
    else:
        # Step 1: Load .img files
        print(f'\n[1/4] Loading {n} files...', end=' ', flush=True)
        t0 = time.time()
        vol = load_unwarp_folder(folder, bin_xy=1, morph_size=morph_arg,
                                 max_workers=args.workers)
        dt = time.time() - t0
        nh, nk, nl = vol.intensity.shape
        ms = vol.metadata['morph_size']
        mode = vol.metadata.get('morph_mode', 'manual')
        s_pp = vol.metadata.get('morph_s_per_pixel')
        recip = vol.metadata.get('morph_recip_period')
        nu_mean = vol.metadata['n_unmeasured_per_frame_mean']
        nu_min = vol.metadata['n_unmeasured_per_frame_min']
        nu_max = vol.metadata['n_unmeasured_per_frame_max']
        nu_pct = vol.metadata['n_unmeasured_per_frame_pct']
        print(f'{dt:.1f}s  ({nh}x{nk}x{nl}, {vol.intensity.dtype})')
        if mode == 'auto' and s_pp is not None and recip is not None:
            print(f'      Mask kernel: morph={ms}px  (auto: '
                  f's={s_pp:.5f} 1/A, |a*|={recip:.4f} 1/A, '
                  f'5% of one Bragg cell)')
        else:
            print(f'      Mask kernel: morph={ms}px  (manual)')
        print(f'      Coverage:    unmeasured/frame mean={nu_mean:,.0f} '
              f'({nu_pct:.2f}%) [min {nu_min:,}, max {nu_max:,}]')

        # Step 2: Save raw
        print(f'[2/4] Saving raw -> {os.path.basename(raw_path)}...',
              end=' ', flush=True)
        t0 = time.time()
        save_volume_h5(raw_path, vol)
        dt = time.time() - t0
        size_mb = os.path.getsize(raw_path) / 1e6
        print(f'{dt:.1f}s  ({size_mb:.0f} MB)')

    # ── Step 3: Bin ──
    if args.bin > 1 or args.binl > 1:
        print(f'[3/4] Binning {args.bin}x{args.bin}x{args.binl}...', end=' ', flush=True)
        vol = bin_native(vol, args.bin, args.binl)
        nh, nk, nl = vol.intensity.shape
        print(f'{nh}x{nk}x{nl}')
    else:
        print(f'[3/4] No binning')

    # ── Step 4: Single-pass symmetrize + outlier rejection ──
    print(f'[4/4] Symmetrize + outlier rejection ({device})...',
          end=' ', flush=True)
    t0 = time.time()
    vol = symmetrize_volume(vol, args.laue, sigma=sigma_val, use_gpu=use_gpu)
    dt_sym = time.time() - t0
    n_removed = vol.metadata.get('n_outliers_removed', 0)
    print(f'{dt_sym:.1f}s  ({n_removed:,} voxels flagged as outliers, '
          f'{vol.metadata["symmetry_mapping"]} index maps)')

    # Save
    sym_path = os.path.join(folder, f'{base}_sym_{laue_clean}.h5')
    print(f'      Saving -> {os.path.basename(sym_path)}...', end=' ', flush=True)
    t0 = time.time()
    save_volume_h5(sym_path, vol)
    dt_save = time.time() - t0
    size_mb = os.path.getsize(sym_path) / 1e6
    print(f'{dt_save:.1f}s  ({size_mb:.0f} MB)')

    print(f'\n  Range: [{np.nanmin(vol.intensity):.3f}, '
          f'{np.nanmax(vol.intensity):.3f}]')
    print(f'  Raw:   {raw_path}')
    print(f'  Sym:   {sym_path}')
    print('  Done.')


if __name__ == '__main__':
    main()
