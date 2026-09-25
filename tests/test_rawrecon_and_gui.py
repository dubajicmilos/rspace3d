"""Regression tests for the rawrecon and interface findings ported into 2.1.

Synthetic, CPU-only. GUI tests run offscreen (QT_QPA_PLATFORM=offscreen is
set below if unset).
"""
from __future__ import annotations

import os
import struct

import numpy as np
import pytest

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from rspace3d.rawrecon import engine, tabbin, calibrate, corrections
from rspace3d.rawrecon.geometry import FlatDetector


# ──────────────────────────────────────────────────────────────────
# D13: frames / angles must pair up
# ──────────────────────────────────────────────────────────────────

def test_mismatched_frames_and_angles_raise():
    with pytest.raises(ValueError, match='frames length 2 != phis length 1'):
        engine._validate_series(['a.cbf', 'b.cbf'], [0.0])
    with pytest.raises(ValueError, match='at least one frame'):
        engine._validate_series([], [])
    frames, phis = engine._validate_series(['a.cbf'], [1.5])
    assert frames == ['a.cbf'] and phis == [1.5]


# ──────────────────────────────────────────────────────────────────
# D12: Bragg registration RMS over finite candidates only
# ──────────────────────────────────────────────────────────────────

def test_bragg_rms_ignores_non_finite_candidates():
    axes = np.arange(-2.0, 2.01, 0.5)
    data = np.full((9, 9, 9), np.nan, np.float32)
    counts = np.zeros((9, 9, 9), np.int64)
    data[7, 4, 4] = 100.0                   # (1.5, 0, 0): sole valid voxel, 0.5 rlu off the node
    counts[7, 4, 4] = 50
    vol = engine.Volume(data, axes, axes, axes, np.eye(3), counts, 0.5)
    assert engine.bragg_registration_rms(vol, K=300) == pytest.approx(0.5)
    counts[:] = 0
    assert np.isnan(engine.bragg_registration_rms(vol))


# ──────────────────────────────────────────────────────────────────
# D3 / D14: rawrecon volumes are regular hkl grids with counts
# ──────────────────────────────────────────────────────────────────

def test_reconstruct_to_volumedata_declares_regular_grid_and_keeps_counts():
    axes = (np.arange(8) - 3.5) * 0.25
    data = np.zeros((8, 8, 8), np.float32)
    counts = np.ones((8, 8, 8), np.int64)
    ub = np.diag([1 / 4.0, 1 / 5.0, 1 / 6.0])   # orthorhombic: a != b != c
    vol = engine.Volume(data, axes, axes, axes, ub, counts, 0.5)
    vd = engine.reconstruct_to_volumedata(vol, source_folder='x', extra_meta={'hot_pixel_cutoff': None})
    assert vd.metadata['grid_kind'] == 'hkl_regular'
    assert not {'M_inv', 's', 'cx', 'cy'} & set(vd.metadata)
    assert vd.counts is counts
    from rspace3d.volume_builder import volume_affine
    origin, A = volume_affine(vd)
    assert np.allclose(A, np.eye(3) * 0.25) and np.allclose(origin, -0.875)


# ──────────────────────────────────────────────────────────────────
# I1: the tabbin next to the par in use wins over an unrelated larger table
# ──────────────────────────────────────────────────────────────────

def test_find_tabbin_prefers_table_next_to_the_par(tmp_path):
    sub = tmp_path / 'processed'; sub.mkdir()
    par = sub / 'run_1_cracker.par'; par.write_text('x')
    own = sub / 'run_1_peakhunt.tabbin'; own.write_bytes(b'0' * 10)
    decoy = tmp_path / 'otherrun_peakhunt.tabbin'; decoy.write_bytes(b'0' * 1000)
    assert tabbin.find_tabbin(str(tmp_path), par_path=str(par)) == str(own)
    own.unlink()
    # a single unrelated table is used as the fallback ...
    assert tabbin.find_tabbin(str(tmp_path), par_path=str(par)) == str(decoy)
    # ... several unrelated tables are ambiguous
    (tmp_path / 'third_peakhunt.tabbin').write_bytes(b'0' * 500)
    with pytest.raises(ValueError, match='ambiguous|pass tab_path'):
        tabbin.find_tabbin(str(tmp_path), par_path=str(par))


# ──────────────────────────────────────────────────────────────────
# S1: pixel-mapping evidence margin
# ──────────────────────────────────────────────────────────────────

class _FakeImage:
    def __init__(self, data):
        self.data = data


def _peaks(n, nx, ny, seed=0):
    rng = np.random.default_rng(seed)
    px = rng.integers(20, nx - 20, n)
    py = rng.integers(20, ny - 20, n)
    return {'px': px, 'py': py, 'intensity': rng.integers(100, 1000, n),
            'frame': np.ones(n, np.int32)}


def test_ground_mapping_refuses_flat_frames_and_accepts_spots(monkeypatch):
    import fabio
    nx, ny = 200, 160
    pk = _peaks(30, nx, ny)
    keep = np.ones(30, bool)
    flat = np.ones((ny, nx), np.int32)
    monkeypatch.setattr(fabio, 'open', lambda path: _FakeImage(flat))
    with pytest.raises(RuntimeError, match='not discriminating'):
        calibrate._ground_mapping(pk, keep, ['f1.cbf'], nx, ny)
    # spots consistent with the identity mapping (col = px, row = py, 1-based)
    img = np.ones((ny, nx), np.int32)
    for x, y in zip(pk['px'], pk['py']):
        img[y - 1, x - 1] = 5000
    monkeypatch.setattr(fabio, 'open', lambda path: _FakeImage(img))
    col, row, name, margin = calibrate._ground_mapping(pk, keep, ['f1.cbf'], nx, ny)
    assert name == '---' and margin >= calibrate.MAPPING_MIN_MARGIN
    assert np.array_equal(col, pk['px']) and np.array_equal(row, pk['py'])


# ──────────────────────────────────────────────────────────────────
# polarisation axis
# ──────────────────────────────────────────────────────────────────

def test_polarisation_axis_perpendicular_to_oscillation_axis_matches_fast_when_aligned():
    det = FlatDetector(distance=85.0, pixel_size=0.3, beam_center=(500, 400),
                       shape=(800, 1000), wavelength=0.5)      # wide angles (I19-2-like)
    osc = np.array([0.0, 1.0, 0.0])
    e_vec = np.cross(det.beam, osc)
    assert np.allclose(np.abs(e_vec), [1, 0, 0])
    default = corrections.pixel_corrections(det)
    explicit = corrections.pixel_corrections(det, horizontal=e_vec)
    assert np.allclose(default, explicit)
    # the orthogonal choice (E along the rotation axis) is a different correction
    other = corrections.pixel_corrections(det, horizontal=osc)
    assert np.abs(other / default - 1).max() > 0.01


# ──────────────────────────────────────────────────────────────────
# D15 / D6: header plane detection and layer validation
# ──────────────────────────────────────────────────────────────────

def _img_header(nx, ny, plane='HK', fixed=0.0, flags=True):
    hdr = bytearray(2400)
    struct.pack_into('<H', hdr, 278, nx); struct.pack_into('<H', hdr, 280, ny)
    off = {'HK': 880, 'HL': 872, 'KL': 864}[plane]
    struct.pack_into('<d', hdr, off, fixed)
    if flags:
        for o in {'HK': (896, 936), 'HL': (896, 944), 'KL': (904, 944)}[plane]:
            struct.pack_into('<d', hdr, o, 1.0)
    struct.pack_into('<d', hdr, 1024, 0.7); struct.pack_into('<d', hdr, 2104, 0.5)
    for i, v in enumerate(np.eye(3).ravel() * 0.1):
        struct.pack_into('<d', hdr, 2256 + 8 * i, v)
    return bytes(hdr)


def test_flagless_header_uses_fixed_value_and_ambiguity_raises(tmp_path):
    from rspace3d.volume_builder import _read_header_fast
    p = tmp_path / 'a_1.img'; p.write_bytes(_img_header(4, 4, 'HL', 1.5, flags=False))
    assert _read_header_fast(str(p))['plane_type'] == 'HL'
    p.write_bytes(_img_header(4, 4, 'HK', 0.0, flags=False))      # nothing set
    with pytest.raises(ValueError, match='Cannot determine plane type'):
        _read_header_fast(str(p))


def test_scan_rejects_duplicate_and_missing_layers(tmp_path):
    from rspace3d.volume_builder import scan_unwarp_folder
    for i, l in enumerate([-0.5, 0.0, 0.5]):
        (tmp_path / f'run_{i + 1}.img').write_bytes(_img_header(4, 4, 'HK', l))
    assert [v for _, v in scan_unwarp_folder(str(tmp_path))] == [-0.5, 0.0, 0.5]
    (tmp_path / 'run_4.img').write_bytes(_img_header(4, 4, 'HK', 0.5))       # duplicate
    with pytest.raises(ValueError, match='Duplicate'):
        scan_unwarp_folder(str(tmp_path))
    (tmp_path / 'run_4.img').write_bytes(_img_header(4, 4, 'HK', 1.5))       # gap at 1.0
    with pytest.raises(ValueError, match='not uniform'):
        scan_unwarp_folder(str(tmp_path))


# ──────────────────────────────────────────────────────────────────
# GUI state: D9, I2, I3
# ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope='module')
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    yield app


def test_builder_gui_guards_unreadable_folder_and_stale_enable(qapp, tmp_path):
    from rspace3d.volume_builder_gui import SimpleVolumeGUI
    gui = SimpleVolumeGUI()
    gui._set_folder(str(tmp_path))                        # no .img files
    assert not gui.process_btn.isEnabled()
    gui._set_busy(False)
    assert not gui.process_btn.isEnabled()                # I3: stays off
    (tmp_path / 'x_1.img').write_bytes(b'\x00' * 32)      # truncated header
    gui._set_folder(str(tmp_path))                        # I2: no exception
    assert not gui._dataset_valid and not gui.process_btn.isEnabled()


def test_reconstruct_gui_clears_previous_volume_on_folder_change(qapp, tmp_path):
    from rspace3d.reconstruct_gui import ReconstructGUI
    gui = ReconstructGUI()
    gui._vd = object(); gui._vol = object(); gui._info = {'name': 'old'}
    gui._set_post_enabled(True)
    gui._set_folder(str(tmp_path))                        # no cbf frames
    assert gui._vd is None and gui._vol is None and gui._info is None
    assert not gui._post_group.isEnabled() and not gui.recon_btn.isEnabled()
    assert gui.hot.value() == 0                           # no hot-pixel clip by default


# ──────────────────────────────────────────────────────────────────
# Loader end-to-end (fabio decode stubbed): D4, D7, D10
# ──────────────────────────────────────────────────────────────────

def _stub_decode(monkeypatch, frames):
    """Serve synthetic (ny, nx) int32 frames instead of fabio-decoded .img data."""
    from rspace3d import volume_builder as vb
    from rspace3d.rsp_reader import RSPLayer
    monkeypatch.setattr(vb, '_read_intensity', lambda path: frames[os.path.basename(path)])

    def fake_layer(path):
        hdr = vb._read_header_fast(path)
        nx, ny = hdr['nx'], hdr['ny']
        s = 2.0 / (hdr['d_min'] * nx)
        return RSPLayer(intensity=None, idx1=None, idx2=None, fixed_value=hdr['fixed_value'],
                        plane_type=hdr['plane_type'], x_label='h', y_label='k', fixed_label='l',
                        thickness=(0, 0), M_inv=np.eye(2) / 0.1, s=s,
                        cx=(nx + 1) // 2 + 0.5, cy=(ny + 1) // 2 + 0.5, step_idx1=1, step_idx2=1)
    monkeypatch.setattr(vb, 'read_rsp_layer', fake_layer)


def test_loader_covered_binning_and_binned_geometry(monkeypatch, tmp_path):
    from rspace3d.volume_builder import load_unwarp_folder
    nx, ny = 12, 12
    frames = {}
    for i, l in enumerate([-0.5, 0.0, 0.5]):
        name = f'run_{i + 1}.img'
        (tmp_path / name).write_bytes(_img_header(nx, ny, 'HK', l))
        f = np.full((ny, nx), 100, np.int32)
        f[:, :3] = -1                     # detector-flagged strip -> unmeasured
        f[4, 4] = 0                        # an isolated real zero count
        frames[name] = f
    _stub_decode(monkeypatch, frames)
    vol = load_unwarp_folder(str(tmp_path), bin_xy=2, morph_size=3, max_workers=1)
    assert vol.plane_type == 'HK' and vol.intensity.shape == (6, 6, 3)
    # D4: the 2x2 cells straddling the -1 strip average the covered pixels only
    assert vol.intensity[1, 2, 0] == 100.0            # x pixels 2,3 -> one covered column
    assert np.isnan(vol.intensity[0, 2, 0])            # fully unmeasured block
    assert vol.intensity[2, 2, 0] == pytest.approx(75.0)   # block with the real zero
    # D10: raster geometry describes the binned volume; bin_z applied once
    assert vol.metadata['s'] == pytest.approx(2 * 2.0 / (0.7 * nx))
    assert vol.metadata['bin_xy'] == 2 and vol.metadata['bin_z'] == 1
    assert vol.metadata['cx'] == pytest.approx(1 - vol.H[0] / (vol.H[1] - vol.H[0]))
    assert 'source_manifest' in vol.metadata
    volz = load_unwarp_folder(str(tmp_path), bin_xy=1, bin_z=3, morph_size=3, max_workers=1)
    assert volz.metadata['bin_z'] == 3 and volz.intensity.shape == (nx, ny, 1)


def test_loader_stores_hl_folder_in_physical_order(monkeypatch, tmp_path):
    from rspace3d.volume_builder import load_unwarp_folder, extract_volume_slice
    nx, ny = 6, 4
    frames = {}
    for i, k in enumerate([0.0, 1.0]):
        name = f'run_{i + 1}.img'
        (tmp_path / name).write_bytes(_img_header(nx, ny, 'HL', k))
        frames[name] = np.full((ny, nx), 10 * (i + 1), np.int32)
    _stub_decode(monkeypatch, frames)
    vol = load_unwarp_folder(str(tmp_path), morph_size=3, max_workers=1)
    assert vol.plane_type == 'HL'
    assert vol.intensity.shape == (nx, 2, ny)            # (h, k, l)
    assert len(vol.H) == nx and list(vol.K) == [0.0, 1.0] and len(vol.L) == ny
    assert np.all(vol.intensity[:, 1, :] == 20)
    sl, x_ax, y_ax, xl, yl, fl, fv, n = extract_volume_slice(vol, 1, 1.0)   # native HL cut
    assert (xl, yl, fl) == ('h', 'l', 'k') and sl.shape == (ny, nx) and np.all(sl == 20)
