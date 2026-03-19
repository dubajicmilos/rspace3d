"""
volume_process.py — Command-line 3D volume processing.

One-line usage:
    python volume_process.py F:\\path\\to\\unwarp
    python volume_process.py F:\\path\\to\\unwarp --laue m-3m --sigma 5 --bin 4 --iter 2

Full pipeline: load -> save raw -> bin -> reject outliers -> symmetrize -> save.

Options:
    --laue    Laue group (default: m-3m)
              Choices: -1, 2/m, mmm, 4/m, 4/mmm, -3, -3m, 6/m, 6/mmm, m-3, m-3m
    --sigma   Outlier rejection sigma (default: 3.0)
    --iter    Rejection iterations (default: 1)
    --bin     HK bin factor (default: 2)
    --binl    L bin factor (default: 1)
    --no-gpu  Force CPU even if GPU available
"""

import argparse
import os
import sys
import time
import re
import numpy as np

from rspace3d.volume_builder import (
    load_unwarp_folder, bin_volume,
    reject_outliers, symmetrize_volume,
    save_volume_h5, find_par_file, read_par_cell, cell_from_ub,
    _read_header_fast, LAUE_GROUP_NAMES, _EXPECTED_ORDERS, HAS_GPU,
)


def main():
    parser = argparse.ArgumentParser(
        description='Process CrysAlisPro unwarp .img files into symmetrized 3D volume.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='Examples:\n'
               '  python volume_process.py F:\\data\\unwarp\n'
               '  python volume_process.py F:\\data\\unwarp --laue 4/mmm --sigma 5 --bin 4\n'
               '  python volume_process.py F:\\data\\unwarp --no-gpu\n')

    parser.add_argument('folder', help='Path to unwarp folder containing .img files')
    parser.add_argument('--laue', default='m-3m',
                        choices=list(LAUE_GROUP_NAMES.keys()),
                        help='Laue group (default: m-3m)')
    parser.add_argument('--sigma', type=float, default=3.0,
                        help='Outlier rejection sigma (default: 3.0)')
    parser.add_argument('--iter', type=int, default=1,
                        help='Rejection iterations (default: 1)')
    parser.add_argument('--bin', type=int, default=2,
                        help='HK bin factor (default: 2)')
    parser.add_argument('--binl', type=int, default=1,
                        help='L bin factor (default: 1)')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Force CPU even if GPU available')

    args = parser.parse_args()
    folder = args.folder
    use_gpu = HAS_GPU and not args.no_gpu

    if not os.path.isdir(folder):
        print(f'Error: {folder} is not a directory')
        sys.exit(1)

    # Count files (filter to single prefix)
    from rspace3d.volume_builder import _filter_numbered_imgs
    img_files = _filter_numbered_imgs(folder)
    if not img_files:
        print(f'Error: no numbered .img files found in {folder}')
        sys.exit(1)

    def _num(f):
        try: return int(f.rsplit('_', 1)[1].split('.')[0])
        except: return 0
    sorted_f = sorted(img_files, key=_num)
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
    par_path = find_par_file(folder)
    cell = None
    if par_path:
        print(f'  Par:     {os.path.basename(par_path)}')
        cell = read_par_cell(par_path)
    if cell is None:
        cell = cell_from_ub(hdr['ub'], hdr['wavelength'])
    if cell:
            print(f'  Cell:    a={cell["a"]:.5f}  b={cell["b"]:.5f}  c={cell["c"]:.5f} A')
            print(f'           alpha={cell["alpha"]:.3f}  beta={cell["beta"]:.3f}  '
                  f'gamma={cell["gamma"]:.3f} deg')

    print(f'  Laue:    {args.laue} ({_EXPECTED_ORDERS[args.laue]} ops)')
    print(f'  Sigma:   {args.sigma}  |  Iterations: {args.iter}')
    print(f'  Binning: {args.bin}x{args.bin} (HK), {args.binl}x (L)')
    print('=' * 60)

    # ── Step 1: Load ──
    print(f'\n[1/4] Loading {n} files...', end=' ', flush=True)
    t0 = time.time()
    vol = load_unwarp_folder(folder, bin_xy=1)
    dt = time.time() - t0
    nh, nk, nl = vol.intensity.shape
    print(f'{dt:.1f}s  ({nh}x{nk}x{nl}, {vol.intensity.dtype})')

    # ── Step 2: Save raw ──
    raw_path = os.path.join(folder, f'{base}_raw.h5')
    print(f'[2/4] Saving raw -> {os.path.basename(raw_path)}...', end=' ', flush=True)
    t0 = time.time()
    save_volume_h5(raw_path, vol)
    dt = time.time() - t0
    size_mb = os.path.getsize(raw_path) / 1e6
    print(f'{dt:.1f}s  ({size_mb:.0f} MB)')

    # ── Step 3: Bin ──
    if args.bin > 1 or args.binl > 1:
        print(f'[3/4] Binning {args.bin}x{args.bin}x{args.binl}...', end=' ', flush=True)
        vol = bin_volume(vol, args.bin, args.bin, args.binl)
        nh, nk, nl = vol.intensity.shape
        print(f'{nh}x{nk}x{nl}')
    else:
        print(f'[3/4] No binning')

    # ── Step 4: Reject + Symmetrize ──
    print(f'[4/4] Outlier rejection ({device})...', end=' ', flush=True)
    t0 = time.time()
    vol = reject_outliers(vol, args.laue, sigma=args.sigma,
                          n_iter=args.iter, use_gpu=use_gpu)
    dt_rej = time.time() - t0
    n_repl = vol.metadata.get('n_outliers_replaced', 0)
    print(f'{dt_rej:.1f}s  ({n_repl:,} replaced)')

    print(f'      Symmetrizing ({args.laue})...', end=' ', flush=True)
    t0 = time.time()
    vol = symmetrize_volume(vol, args.laue, use_gpu=use_gpu)
    dt_sym = time.time() - t0
    print(f'{dt_sym:.1f}s')

    # Save
    sym_path = os.path.join(folder, f'{base}_sym_{laue_clean}.h5')
    print(f'      Saving -> {os.path.basename(sym_path)}...', end=' ', flush=True)
    t0 = time.time()
    save_volume_h5(sym_path, vol)
    dt_save = time.time() - t0
    size_mb = os.path.getsize(sym_path) / 1e6
    print(f'{dt_save:.1f}s  ({size_mb:.0f} MB)')

    print(f'\n  Range: [{vol.intensity.min():.1f}, {vol.intensity.max():.1f}]')
    print(f'  Raw:   {raw_path}')
    print(f'  Sym:   {sym_path}')
    print('  Done.')


if __name__ == '__main__':
    main()
