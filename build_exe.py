"""
Build standalone executables using PyInstaller.

Usage:
    pip install pyinstaller
    python build_exe.py

Creates dist/ folder with:
    rspace3d-viewer.exe   — Unified viewer (single .img / .h5 volumes)
    rspace3d-builder.exe  — One-button volume processor GUI
    rspace3d-process.exe  — Command-line processor
"""

import subprocess
import sys

APPS = [
    {
        'name': 'rspace3d-viewer',
        'script': 'rspace3d/rsp_viewer.py',
        'icon': None,
        'console': False,
    },
    {
        'name': 'rspace3d-builder',
        'script': 'rspace3d/volume_builder_gui.py',
        'icon': None,
        'console': False,
    },
    {
        'name': 'rspace3d-process',
        'script': 'rspace3d/volume_process.py',
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
        # fabio registers image-format plugins at import time via importlib,
        # which PyInstaller's static analysis can't see. Without this the
        # bundle crashes on startup with ModuleNotFoundError for e.g.
        # fabio.pilatusimage / fabio.brukerimage / fabio.cbfimage.
        '--collect-submodules', 'fabio',
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
