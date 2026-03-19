"""
volume_isosurface.py — 3D isosurface visualization for reciprocal space volumes.

Uses Plotly (available) for interactive browser-based 3D rendering.
Optionally uses PyVista (if installed) for higher-quality VTK rendering.

Usage:
    from .volume_isosurface import plot_isosurface, plot_isosurface_pyvista
    from volume_builder import load_volume_h5

    vol = load_volume_h5('sample_sym_mbar3m.h5')
    plot_isosurface(vol, isovalue=50)                    # Plotly (browser)
    plot_isosurface(vol, isovalue=50, method='pyvista')  # PyVista (VTK)

    # Or from command line:
    python volume_isosurface.py sample.h5 --iso 50
"""

import numpy as np
from .volume_builder import VolumeData, load_volume_h5


def plot_isosurface(vol, isovalue=None, method='plotly',
                    colormap='plasma', opacity=0.6,
                    h_range=None, k_range=None, l_range=None,
                    title=None, save_html=None):
    """Plot 3D isosurface of a reciprocal space volume.

    Parameters
    ----------
    vol : VolumeData or str
        Volume data or path to .h5 file.
    isovalue : float or list of float
        Isosurface level(s). If None, uses 50th percentile of nonzero data.
    method : str
        'plotly' (default, browser-based) or 'pyvista' (VTK, higher quality).
    colormap : str
        Colormap name.
    opacity : float
        Surface opacity (0-1).
    h_range, k_range, l_range : tuple (min, max)
        Crop to subvolume before plotting.
    title : str
        Plot title.
    save_html : str
        Save Plotly figure as HTML file (Plotly only).
    """
    if isinstance(vol, str):
        vol = load_volume_h5(vol)

    data, H, K, L = _prepare_data(vol, h_range, k_range, l_range)

    if isovalue is None:
        nz = data[data > 0]
        isovalue = float(np.percentile(nz, 50)) if len(nz) > 0 else 1.0

    if title is None:
        cell = vol.metadata.get('cell')
        laue = vol.metadata.get('laue_group', '')
        title = f'Isosurface at I={isovalue:.0f}'
        if laue:
            title += f' ({laue})'

    if method == 'pyvista':
        _plot_pyvista(data, H, K, L, isovalue, colormap, opacity, title)
    else:
        _plot_plotly(data, H, K, L, isovalue, colormap, opacity, title, save_html)


def _prepare_data(vol, h_range, k_range, l_range):
    """Crop volume to specified Miller index ranges."""
    data = vol.intensity.astype(np.float32)
    H, K, L = vol.H, vol.K, vol.L

    if h_range:
        mask = (H >= h_range[0]) & (H <= h_range[1])
        data = data[mask, :, :]
        H = H[mask]
    if k_range:
        mask = (K >= k_range[0]) & (K <= k_range[1])
        data = data[:, mask, :]
        K = K[mask]
    if l_range:
        mask = (L >= l_range[0]) & (L <= l_range[1])
        data = data[:, :, mask]
        L = L[mask]

    return data, H, K, L


def _plot_plotly(data, H, K, L, isovalue, colormap, opacity, title, save_html):
    """Plotly-based isosurface (interactive, browser)."""
    import plotly.graph_objects as go

    # Create meshgrid for Plotly (needs flat arrays)
    HH, KK, LL = np.meshgrid(H, K, L, indexing='ij')

    isovalues = isovalue if isinstance(isovalue, (list, tuple)) else [isovalue]

    fig = go.Figure()

    for iso in isovalues:
        fig.add_trace(go.Isosurface(
            x=HH.flatten(),
            y=KK.flatten(),
            z=LL.flatten(),
            value=data.flatten(),
            isomin=iso,
            isomax=iso,
            surface_count=1,
            opacity=opacity,
            colorscale=colormap,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='h',
            yaxis_title='k',
            zaxis_title='l',
            aspectmode='data',
        ),
        width=900, height=700,
    )

    if save_html:
        fig.write_html(save_html)
        print(f'Saved: {save_html}')

    fig.show()


def _plot_pyvista(data, H, K, L, isovalue, colormap, opacity, title):
    """PyVista-based isosurface with interactive controls.

    Features:
    - Isovalue slider (log scale for better range selection)
    - Symmetric clipping sliders per axis (clips to [-v, +v])
    - Clips preserved when isovalue changes
    - Optimised: uint16 data if possible for lower memory
    """
    try:
        import pyvista as pv
    except ImportError:
        print("PyVista not installed. Install with: pip install pyvista")
        print("Falling back to Plotly...")
        _plot_plotly(data, H, K, L, isovalue, colormap, opacity, title, None)
        return

    from scipy.ndimage import gaussian_filter

    # Light smoothing
    data_smooth = gaussian_filter(data.astype(np.float32), sigma=0.5)

    # Convert to uint16 if possible (halves memory, VTK supports it)
    dmax = data_smooth.max()
    if dmax > 0 and dmax < 65535 and data_smooth.min() >= 0:
        scale = 65535.0 / dmax
        grid_data = (data_smooth * scale).astype(np.uint16)
        use_scaled = True
    else:
        grid_data = data_smooth.astype(np.float32)
        scale = 1.0
        use_scaled = False

    # Build ImageData grid
    dh = float(H[1] - H[0]) if len(H) > 1 else 1.0
    dk = float(K[1] - K[0]) if len(K) > 1 else 1.0
    dl = float(L[1] - L[0]) if len(L) > 1 else 1.0

    grid = pv.ImageData(
        dimensions=(len(H), len(K), len(L)),
        spacing=(abs(dh), abs(dk), abs(dl)),
        origin=(float(H[0]), float(K[0]), float(L[0])),
    )
    grid['intensity'] = grid_data.flatten(order='F')

    # Intensity stats for slider range (in original units)
    nz = data_smooth[data_smooth > 0]
    if len(nz) == 0:
        print("No nonzero data to display")
        return

    # Start high (99th percentile) to avoid noise, user can lower
    iso_start = float(np.percentile(nz, 99))
    iso_min_log = max(np.log10(max(float(np.percentile(nz, 50)), 0.5)), -1)
    iso_max_log = np.log10(max(float(np.percentile(nz, 99.99)), 1.0))

    if isovalue is not None:
        isovalues = isovalue if isinstance(isovalue, (list, tuple)) else [isovalue]
        initial_iso = isovalues[0]
    else:
        initial_iso = iso_start

    # Shared state
    state = {
        'iso': initial_iso,
        'h_clip': max(abs(float(H[0])), abs(float(H[-1]))),
        'k_clip': max(abs(float(K[0])), abs(float(K[-1]))),
        'l_clip': max(abs(float(L[0])), abs(float(L[-1]))),
    }
    h_max = state['h_clip']
    k_max = state['k_clip']
    l_max = state['l_clip']

    def _rebuild():
        """Rebuild isosurface with current state."""
        iso = state['iso']
        iso_scaled = iso * scale if use_scaled else iso

        # Symmetric clip: keep data within [-clip, +clip] for each axis
        clipped = grid
        if state['h_clip'] < h_max - 0.01:
            clipped = clipped.clip(normal='x', origin=(state['h_clip'], 0, 0), invert=True)
            clipped = clipped.clip(normal='-x', origin=(-state['h_clip'], 0, 0), invert=True)
        if state['k_clip'] < k_max - 0.01:
            clipped = clipped.clip(normal='y', origin=(0, state['k_clip'], 0), invert=True)
            clipped = clipped.clip(normal='-y', origin=(0, -state['k_clip'], 0), invert=True)
        if state['l_clip'] < l_max - 0.01:
            clipped = clipped.clip(normal='z', origin=(0, 0, state['l_clip']), invert=True)
            clipped = clipped.clip(normal='-z', origin=(0, 0, -state['l_clip']), invert=True)

        pl.remove_actor('isosurface')
        try:
            contour = clipped.contour([iso_scaled], scalars='intensity')
            if contour.n_points > 0:
                pl.add_mesh(contour, opacity=opacity, cmap=colormap,
                            smooth_shading=True, show_scalar_bar=False,
                            name='isosurface')
        except Exception:
            pass

    pl = pv.Plotter()
    pl.set_background('white')

    # Initial render
    iso_scaled = initial_iso * scale if use_scaled else initial_iso
    contour = grid.contour([iso_scaled], scalars='intensity')
    if contour.n_points > 0:
        pl.add_mesh(contour, opacity=opacity, cmap=colormap,
                    smooth_shading=True, show_scalar_bar=False,
                    name='isosurface')

    slider_kw = dict(style='modern', title_height=0.015, tube_width=0.003)

    # --- Isovalue slider (log scale) ---
    initial_log = np.log10(max(initial_iso, 0.1))

    def update_iso_log(log_val):
        state['iso'] = 10 ** log_val
        _rebuild()

    pl.add_slider_widget(
        update_iso_log,
        rng=[iso_min_log, iso_max_log],
        value=initial_log,
        title='log10(Isovalue)',
        pointa=(0.02, 0.94), pointb=(0.30, 0.94),
        **slider_kw,
    )

    # --- Symmetric clipping sliders ---
    def update_clip_h(v):
        state['h_clip'] = v; _rebuild()
    def update_clip_k(v):
        state['k_clip'] = v; _rebuild()
    def update_clip_l(v):
        state['l_clip'] = v; _rebuild()

    pl.add_slider_widget(
        update_clip_h, rng=[0.5, h_max], value=h_max,
        title='h range', pointa=(0.02, 0.05), pointb=(0.22, 0.05),
        **slider_kw,
    )
    pl.add_slider_widget(
        update_clip_k, rng=[0.5, k_max], value=k_max,
        title='k range', pointa=(0.30, 0.05), pointb=(0.52, 0.05),
        **slider_kw,
    )
    pl.add_slider_widget(
        update_clip_l, rng=[0.5, l_max], value=l_max,
        title='l range', pointa=(0.58, 0.05), pointb=(0.78, 0.05),
        **slider_kw,
    )

    # --- Bounding box with Miller index grid ---
    # Box outline
    h_lo, h_hi = float(H[0]), float(H[-1])
    k_lo, k_hi = float(K[0]), float(K[-1])
    l_lo, l_hi = float(L[0]), float(L[-1])
    bounds = [h_lo, h_hi, k_lo, k_hi, l_lo, l_hi]
    box = pv.Box(bounds)
    pl.add_mesh(box, style='wireframe', color='gray', opacity=0.3,
                line_width=1, name='bbox')

    # Grid lines at integer Miller indices on the bounding box faces
    grid_color = 'gray'
    grid_opacity = 0.15
    grid_width = 0.5

    # h-grid lines on the k-l faces
    for h in range(int(np.ceil(h_lo)), int(np.floor(h_hi)) + 1):
        for face_k in [k_lo, k_hi]:
            line = pv.Line((h, face_k, l_lo), (h, face_k, l_hi))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)
        for face_l in [l_lo, l_hi]:
            line = pv.Line((h, k_lo, face_l), (h, k_hi, face_l))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)

    # k-grid lines on the h-l faces
    for k in range(int(np.ceil(k_lo)), int(np.floor(k_hi)) + 1):
        for face_h in [h_lo, h_hi]:
            line = pv.Line((face_h, k, l_lo), (face_h, k, l_hi))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)
        for face_l in [l_lo, l_hi]:
            line = pv.Line((h_lo, k, face_l), (h_hi, k, face_l))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)

    # l-grid lines on the h-k faces
    for l in range(int(np.ceil(l_lo)), int(np.floor(l_hi)) + 1):
        for face_h in [h_lo, h_hi]:
            line = pv.Line((face_h, k_lo, l), (face_h, k_hi, l))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)
        for face_k in [k_lo, k_hi]:
            line = pv.Line((h_lo, face_k, l), (h_hi, face_k, l))
            pl.add_mesh(line, color=grid_color, opacity=grid_opacity,
                        line_width=grid_width)

    pl.add_axes(xlabel='h', ylabel='k', zlabel='l',
                line_width=2, label_size=(0.15, 0.05))
    pl.camera_position = 'iso'
    pl.show()


def plot_isosurface_notebook(vol, isovalue=None,
                              h_range=None, k_range=None, l_range=None,
                              colormap='plasma', opacity=0.6, title=None):
    """Plotly isosurface optimized for Jupyter notebooks.

    Returns a Plotly Figure object for inline display.
    """
    if isinstance(vol, str):
        vol = load_volume_h5(vol)

    data, H, K, L = _prepare_data(vol, h_range, k_range, l_range)

    if isovalue is None:
        nz = data[data > 0]
        isovalue = float(np.percentile(nz, 50)) if len(nz) > 0 else 1.0

    if title is None:
        title = f'Isosurface at I={isovalue:.0f}'

    import plotly.graph_objects as go

    HH, KK, LL = np.meshgrid(H, K, L, indexing='ij')
    isovalues = isovalue if isinstance(isovalue, (list, tuple)) else [isovalue]

    fig = go.Figure()
    for iso in isovalues:
        fig.add_trace(go.Isosurface(
            x=HH.flatten(), y=KK.flatten(), z=LL.flatten(),
            value=data.flatten(),
            isomin=iso, isomax=iso,
            surface_count=1, opacity=opacity, colorscale=colormap,
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
        ))

    fig.update_layout(
        title=title,
        scene=dict(xaxis_title='h', yaxis_title='k', zaxis_title='l',
                    aspectmode='data'),
        width=800, height=600,
    )
    return fig


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='3D isosurface of reciprocal space volume')
    parser.add_argument('file', help='Path to .h5 volume file')
    parser.add_argument('--iso', type=float, default=None, help='Isosurface level')
    parser.add_argument('--hrange', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--krange', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--lrange', type=float, nargs=2, default=None, metavar=('MIN', 'MAX'))
    parser.add_argument('--method', choices=['plotly', 'pyvista'], default='plotly')
    parser.add_argument('--opacity', type=float, default=0.6)
    parser.add_argument('--cmap', default='plasma')
    parser.add_argument('--save', default=None, help='Save as HTML (Plotly only)')

    args = parser.parse_args()
    plot_isosurface(
        args.file, isovalue=args.iso, method=args.method,
        colormap=args.cmap, opacity=args.opacity,
        h_range=args.hrange, k_range=args.krange, l_range=args.lrange,
        save_html=args.save)
