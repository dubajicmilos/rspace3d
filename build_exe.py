"""
Build standalone executables using PyInstaller.

Usage:
    pip install pyinstaller
    python build_exe.py

Creates dist/ folder with:
    rspace3d-viewer.exe    — Unified viewer (single .img / .h5 volumes)
    rspace3d-processor.exe — One-button volume processor
    rspace3d-process.exe   — Command-line processor
"""

import subprocess
import sys

APPS = [
    {
        'name': 'rspace3d-viewer',
        'script': 'rspace3d/rsp_unified_viewer.py',
        'icon': None,
        'console': False,
    },
    {
        'name': 'rspace3d-processor',
        'script': 'rspace3d/volume_gui_simple.py',
        'icon': None,
        'console': False,
    },
    {
        'name': 'rspace3d-process',
        'script': 'scripts/volume_process.py',
        'icon': None,
        'console': True,
    },
]

HIDDEN_IMPORTS = [
    'fabio', 'fabio.OXDimage', 'h5py', 'scipy.ndimage',
    'matplotlib.backends.backend_qtagg',
]

for app in APPS:
    cmd = [
        sys.executable, '-m', 'PyInstaller',
        '--onefile',
        f'--name={app["name"]}',
    ]

    if not app['console']:
        cmd.append('--windowed')

    for hi in HIDDEN_IMPORTS:
        cmd.extend(['--hidden-import', hi])

    if app['icon']:
        cmd.extend(['--icon', app['icon']])

    cmd.append(app['script'])

    print(f'\n{"="*60}')
    print(f'Building {app["name"]}...')
    print(f'{"="*60}')
    subprocess.run(cmd, check=True)

print(f'\n{"="*60}')
print('Done! Executables are in dist/')
print('="*60"')
