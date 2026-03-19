"""
Generate CrysAlisPro .dcunwarp files for HK-plane reciprocal space reconstructions.

Creates layer lists for the 'dc unwarp' command. If the total number of layers
exceeds 500 (CrysAlisPro limit), splits into multiple files that continue
sequentially.

Usage:
    python make_dcunwarp.py                     # default: l=-6 to 6, step=0.02, res=0.8
    python make_dcunwarp.py -6 6 0.02 0.8       # explicit: lmin lmax lstep resolution
    python make_dcunwarp.py -10 10 0.01 0.7     # finer grid, better resolution

Output:
    dcunwarp_001.dcunwarp                       (layers 1-500)
    dcunwarp_002.dcunwarp                       (layers 501-1000, if needed)
    ...
"""

import numpy as np
import sys
from datetime import datetime

MAX_LAYERS_PER_FILE = 500


def generate_dcunwarp(l_min, l_max, l_step, resolution=0.8, prefix='dcunwarp'):
    """Generate .dcunwarp file(s) for HK planes at varying l.

    Parameters
    ----------
    l_min : float
        Starting l value (most negative).
    l_max : float
        Ending l value (most positive).
    l_step : float
        Step size in l (r.l.u.).
    resolution : float
        Resolution limit d_min in Angstroms.
    prefix : str
        Output filename prefix.

    Returns
    -------
    filenames : list of str
        Generated filenames.
    """
    # Generate l values
    l_values = np.arange(l_min, l_max + l_step * 0.5, l_step)
    l_values = np.round(l_values, 6)
    n_total = len(l_values)

    print(f"Total layers: {n_total}")
    print(f"l range: [{l_values[0]:.4f}, {l_values[-1]:.4f}], step = {l_step}")

    # Split into chunks of MAX_LAYERS_PER_FILE
    n_files = int(np.ceil(n_total / MAX_LAYERS_PER_FILE))
    if n_files > 1:
        print(f"Splitting into {n_files} files (max {MAX_LAYERS_PER_FILE} per file)")

    timestamp = datetime.now().strftime("%a %b %d %H:%M:%S %Y")

    filenames = []
    for file_idx in range(n_files):
        start = file_idx * MAX_LAYERS_PER_FILE
        end = min(start + MAX_LAYERS_PER_FILE, n_total)
        chunk = l_values[start:end]

        if n_files == 1:
            fname = f"{prefix}.dcunwarp"
        else:
            fname = f"{prefix}_{file_idx + 1:03d}.dcunwarp"

        with open(fname, 'w', newline='\r\n') as f:
            # Header
            f.write(f"# OD layer list for dc unwarp {timestamp} \n")
            f.write("#name   Osx Osy Osz Oex Oey Oez L1x L1y L1z L2x L2y L2z "
                    "res  intunits iscartsian lauecode [islayermirror islayerinversion]\n")
            f.write("#name   Osx Osy Osz L1x L1y L1z L2x L2y L2z "
                    "res  iscartsian lauecode [islayermirror islayerinversion]\n")

            # HK plane: origin at (0, 0, l), axes along (1,0,0) and (0,1,0)
            for i, l_val in enumerate(chunk, start=1):
                f.write(
                    f'"{i}"   '
                    f'{0.0:10.6f}   {0.0:10.6f}  {l_val:10.6f}     '
                    f'{1.0:10.6f}   {0.0:10.6f}   {0.0:10.6f}     '
                    f'{0.0:10.6f}   {1.0:10.6f}   {0.0:10.6f}        '
                    f'{resolution:10.6f}       0       0\n'
                )

        filenames.append(fname)
        print(f"  {fname}: {len(chunk)} layers, "
              f"l = [{chunk[0]:.4f}, {chunk[-1]:.4f}]")

    return filenames


if __name__ == '__main__':
    if len(sys.argv) >= 4:
        l_min = float(sys.argv[1])
        l_max = float(sys.argv[2])
        l_step = float(sys.argv[3])
        resolution = float(sys.argv[4]) if len(sys.argv) >= 5 else 0.8
    else:
        l_min = -6.0
        l_max = 6.0
        l_step = 0.02
        resolution = 0.8

    generate_dcunwarp(l_min, l_max, l_step, resolution)
