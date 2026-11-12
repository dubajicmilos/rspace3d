"""Check the shader of the 3D view against the CPU ray march. Run by test_view3d.py.

The script needs a real display (not the Qt platform "offscreen") and OpenGL 4.1.
It draws a small volume with a triclinic cell and a sheared raster, with voxels
that are not measured, and compares for each tested pixel

* the distance of the first surface along the ray (shader, test mode) with
  `view3d.first_hit`,
* the largest value along the ray (shader, test mode) with `view3d.ray_max`.

Usage: python gl_selftest.py OUT.json      exit code 77: no OpenGL here
"""
import json
import math
import sys

import numpy as np


def make_volume():
    from rspace3d.volume_builder import VolumeData
    nh, nk, nl = 40, 48, 56
    H = np.linspace(-2.0, 1.9, nh)
    K = np.linspace(-2.4, 2.3, nk)
    L = np.linspace(-1.4, 1.35, nl)
    hh, kk, ll = np.meshgrid(H, K, L, indexing='ij')
    data = 5.0 + 0.5 * np.sin(3.0 * hh) * np.cos(2.0 * kk + ll)
    for (h0, k0, l0, amp, width) in ((0.0, 0.0, 0.0, 4.0e4, 0.12), (1.0, -1.0, 0.5, 6.0e3, 0.2),
                                     (-1.0, 1.0, -0.5, 9.0e2, 0.3), (0.5, 1.5, 1.0, 2.0e2, 0.45),
                                     (-1.5, -1.8, -1.0, 3.0e3, 0.25)):
        data += amp * np.exp(-((hh - h0) ** 2 + (kk - k0) ** 2 + (ll - l0) ** 2) / (2.0 * width ** 2))
    data = data.astype(np.float32)
    data[22:25, :, :] = np.nan                       # a gap, as between detector modules
    data[:, :4, :10] = 0.0                           # zero means "not measured" in old files
    meta = {'grid_kind': 'unwarp_raster', 'M_inv': np.array([[6.0, -0.45], [0.0, 7.0]]),
            'cell': {'a': 6.0, 'b': 7.0, 'c': 9.0, 'alpha': 82.0, 'beta': 104.0, 'gamma': 95.0}}
    return VolumeData(intensity=data, H=H, K=K, L=L, plane_type='HK', metadata=meta)


def check_mouse(app, win, settle):
    """Send mouse and wheel events to the canvas and record what the camera did."""
    from PyQt6.QtCore import QEvent, QPoint, QPointF, Qt
    from PyQt6.QtGui import QMouseEvent, QWheelEvent
    canvas, cam = win.canvas, win.camera
    win.surface_radio.setChecked(True)
    win._look('home')
    settle()
    none, left, right = Qt.MouseButton.NoButton, Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton
    no_mod, shift = Qt.KeyboardModifier.NoModifier, Qt.KeyboardModifier.ShiftModifier

    def mouse(kind, x, y, button, buttons, mods=no_mod):
        pos = QPointF(x, y)
        app.sendEvent(canvas, QMouseEvent(kind, pos, canvas.mapToGlobal(pos), button, buttons, mods))

    def drag(button, mods, dx, dy):
        mouse(QEvent.Type.MouseButtonPress, 300, 300, button, button, mods)
        mouse(QEvent.Type.MouseMove, 300 + dx, 300 + dy, none, button, mods)
        mouse(QEvent.Type.MouseButtonRelease, 300 + dx, 300 + dy, button, none, mods)

    out = {}
    yaw, pitch = cam.yaw, cam.pitch
    drag(left, no_mod, 50, 20)
    out['turn'] = [cam.yaw - yaw, cam.pitch - pitch, float(abs(cam.target).max())]
    yaw = cam.yaw
    drag(right, no_mod, 30, -10)
    out['move_right_button'] = [cam.yaw - yaw, float(abs(cam.target).max())]
    target = cam.target.copy()
    drag(left, shift, -30, 10)
    out['move_shift'] = [cam.yaw - yaw, float(abs(cam.target - target).max())]
    zoom = cam.zoom
    pos = QPointF(canvas.width() / 2, canvas.height() / 2)
    app.sendEvent(canvas, QWheelEvent(pos, canvas.mapToGlobal(pos), QPoint(0, 0), QPoint(0, 120), none, no_mod,
                                      Qt.ScrollPhase.NoScrollPhase, False))
    out['zoom_factor'] = cam.zoom / zoom
    # the read-out below the mouse: find a pixel on a surface
    win._look('home')
    settle()
    hit = None
    for y in range(40, canvas.height(), 25):
        for x in range(40, canvas.width(), 25):
            p = canvas.point_at(x, y)
            if p is not None and not p.get('cap'):
                hit = (x, y, p)
                break
        if hit:
            break
    if hit:
        mouse(QEvent.Type.MouseMove, hit[0], hit[1], none, none)
        canvas._emit_hover()
        out['readout_on_surface'] = win.readout.text()
        out['value_over_level'] = hit[2]['value'] / win.scene.sorted_levels()[0].value
    mouse(QEvent.Type.MouseMove, 3, 3, none, none)
    canvas._emit_hover()
    out['readout_off_surface'] = win.readout.text()
    settle()
    return out


def check_window(app, vol, vg):
    """Draw the real window (not on the screen) and look at its image."""
    import time
    from PyQt6.QtCore import Qt
    from rspace3d.view3d_window import View3DWindow
    win = View3DWindow(vol, title='selftest')
    win.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    win.resize(940, 640)
    win.show()

    def settle():
        for _ in range(30):
            app.processEvents()
            time.sleep(0.004)

    settle()
    out = {'canvas': win.canvas is not None, 'problem': win.problem.text()}
    if win.canvas is not None:
        first = vg.image_array(win.image(1.0))
        win.camera.orbit(120.0, 40.0)                # as a drag of the mouse
        win.canvas.refresh()
        settle()
        second = vg.image_array(win.image(1.0))
        win.projection_radio.setChecked(True)
        settle()
        third = vg.image_array(win.image(1.0))
        point = win.canvas.point_at(win.canvas.width() / 2, win.canvas.height() / 2)
        out['mouse'] = check_mouse(app, win, settle)
        out.update({
            'renderer': str(win.canvas.info.get('renderer')), 'error': win.canvas.error,
            'image': [int(first.shape[1]), int(first.shape[0])],
            'colours_surface': int(len(np.unique(first.reshape(-1, 4)[::7], axis=0))),
            'changed_by_turn': float((first != second).any(axis=2).mean()),
            'changed_by_mode': float((second != third).any(axis=2).mean()),
            'point_at_centre': None if point is None else [float(v) for v in point['hkl']],
        })
        # Qt makes a new OpenGL context when the canvas goes to another window: it must draw again
        from PyQt6.QtWidgets import QVBoxLayout, QWidget
        other = QWidget()
        other.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
        other.resize(640, 480)
        QVBoxLayout(other).addWidget(win.canvas)
        other.show()
        settle()
        again = vg.image_array(win.canvas.grabFramebuffer())
        out['after_new_context'] = {'error': win.canvas.error,
                                    'not_background': float((again[..., :3].min(axis=2) < 250).mean())}
        other.close()
    win.close()
    settle()
    return out


def main(out_path):
    from PyQt6.QtWidgets import QApplication
    from rspace3d import ui
    ui.prepare_opengl()                              # as the windows of the package do
    app = QApplication(sys.argv[:1])
    from rspace3d import view3d as v3
    from rspace3d import view3d_gl as vg
    try:
        off = vg.Offscreen()
    except RuntimeError as e:
        print('no OpenGL:', e)
        return 77
    vol = make_volume()
    grid = v3.make_grid(vol)
    data = vol.intensity
    off.load(data, grid)
    width, height = 200, 150
    res = {'info': off.info(), 'cases': []}
    views = [('ortho', v3.Camera()),
             ('perspective', v3.Camera(yaw=1.1, pitch=-0.5, zoom=1.3, ortho=False)),
             ('box', v3.Camera(yaw=2.4, pitch=0.9, zoom=1.6))]
    worst = {'hit_noskip': 0.0, 'hit_skip': 0.0, 'mismatch_noskip': 0.0, 'mismatch_skip': 0.0, 'max': 0.0}
    for name, cam in views:
        lo, hi = ((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)) if name != 'box' else ((0.1, 0.2, 0.15), (0.9, 0.7, 0.8))
        for level in (30.0, 400.0):
            scene = v3.Scene(levels=[v3.Level(level)], lo=lo, hi=hi, debug=1, floor=1.0e-3)
            for skip in (False, True):
                scene.skip = skip
                dist = vg.unpack24(vg.image_array(off.render(scene, cam, width, height))) * vg.t_scale(grid, cam)
                err, miss, n = [], 0, 0
                for py in range(2, height, 5):
                    for px in range(2, width, 5):
                        o, d = cam.ray(grid, px + 0.5, py + 0.5, width, height)
                        ref = v3.first_hit(data, grid, o, d, level, lo, hi, 1.0, scene.floor, cam.ortho)
                        got = dist[py, px]
                        n += 1
                        if (ref is None) != bool(np.isnan(got)):
                            miss += 1
                        elif ref is not None:
                            voxel = 1.0 / float(np.linalg.norm((grid.w2t[:3, :3] @ d) * np.array(grid.shape[::-1])))
                            err.append(abs(got - ref['t']) / voxel)
                case = {'view': name, 'level': level, 'skip': skip, 'pixels': n, 'hits': len(err),
                        'mismatch': miss / n, 'max_error_voxels': max(err) if err else 0.0,
                        'median_error_voxels': float(np.median(err)) if err else 0.0}
                res['cases'].append(case)
                key = 'skip' if skip else 'noskip'
                worst['hit_' + key] = max(worst['hit_' + key], case['max_error_voxels'])
                worst['mismatch_' + key] = max(worst['mismatch_' + key], case['mismatch'])
        # largest value along the ray
        clim = (1.0, 1.0e5)
        scene = v3.Scene(mode='projection', lo=lo, hi=hi, clim=clim, debug=1, jitter=0.0, floor=1.0e-3, skip=False)
        for skip in (False, True):
            scene.skip = skip
            c = vg.unpack24(vg.image_array(off.render(scene, cam, width, height)))
            err = []
            for py in range(2, height, 5):
                for px in range(2, width, 5):
                    o, d = cam.ray(grid, px + 0.5, py + 0.5, width, height)
                    ref = v3.ray_max(data, grid, o, d, lo, hi, 1.0, cam.ortho)
                    if ref is None or np.isnan(c[py, px]):
                        continue
                    want = (math.log(max(ref['value'], scene.floor)) - math.log(clim[0])) / math.log(clim[1] / clim[0])
                    err.append(abs(c[py, px] - min(max(want, 0.0), 1.0)))
            res['cases'].append({'view': name, 'mode': 'projection', 'skip': skip, 'rays': len(err),
                                 'max_error': max(err) if err else 0.0})
            worst['max'] = max(worst['max'], max(err) if err else 0.0)
    # a normal picture must have the surface colour and the background
    scene = v3.Scene(levels=[v3.Level(400.0, (0.12, 0.37, 0.66))])
    rgba = vg.image_array(off.render(scene, v3.Camera(), width, height))
    res['picture'] = {'white_fraction': float((rgba[..., :3].min(axis=2) > 250).mean()),
                      'blue_fraction': float((rgba[..., 2].astype(int) > rgba[..., 0].astype(int) + 30).mean())}
    res['frame_ms'] = off.frame_ms(scene, v3.Camera(), 640, 480)
    res['worst'] = worst
    off.close()
    res['window'] = check_window(app, vol, vg)
    with open(out_path, 'w') as f:
        json.dump(res, f, indent=1)
    print(json.dumps(worst), json.dumps(res['picture']))
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1]))
