"""
view3d_gl.py - the GPU part of the 3D view.

One fragment shader sends a ray through the volume for each pixel. The volume
is a float 3D texture with the intensity as it is in the file. The shader draws

* up to three surfaces of equal intensity, each with a colour and an opacity.
  A surface is where the ray first reaches the level. Its place between two
  samples is found on a logarithmic scale. A surface at the edge of the
  measured data is grey, and a region that the box cuts has a flat cap;
* or the largest intensity along each ray (projection) through a colour map;
* the grid of the whole Miller indices on the far faces of the box.

A second small texture has the largest value of each block of 8^3 voxels, so
that a ray can jump over empty blocks. Nothing is calculated on the CPU when
the level, the box or the view changes.

The module uses the OpenGL classes of PyQt6 only (OpenGL 4.1 core profile).
`RayCaster` holds the textures and draws. `View3DCanvas` is the widget.
`Offscreen` draws images without a window (tests, scripts).
"""

from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import numpy.typing as npt
from PyQt6.QtCore import QPointF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import (QGuiApplication, QImage, QMatrix4x4, QOffscreenSurface, QOpenGLContext,
                         QSurfaceFormat, QVector2D, QVector3D, QVector4D)
from PyQt6.QtOpenGL import (QOpenGLFramebufferObject, QOpenGLShader, QOpenGLShaderProgram,
                            QOpenGLTexture, QOpenGLVersionFunctionsFactory, QOpenGLVersionProfile,
                            QOpenGLVertexArrayObject)
from PyQt6.QtOpenGLWidgets import QOpenGLWidget

from . import view3d as v3

GL_TRIANGLES = 0x0004
GL_COLOR_BUFFER_BIT = 0x4000
GL_LINEAR = 0x2601
GL_FRAMEBUFFER, GL_READ_FRAMEBUFFER, GL_DRAW_FRAMEBUFFER = 0x8D40, 0x8CA8, 0x8CA9
GL_OUT_OF_MEMORY = 0x0505
GL_UNPACK_ALIGNMENT = 0x0CF5
GL_RENDERER, GL_VERSION, GL_MAX_3D_TEXTURE_SIZE = 0x1F01, 0x1F02, 0x8073

FRAME_MS = 33.0                # a frame that takes longer is drawn with fewer pixels while the view moves
SCALES = (1.0, 0.75, 0.5, 0.35, 0.25)      # part of the pixels along each side

VERTEX = """#version 410 core
out vec2 vNdc;
void main() {
    vec2 xy = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    vNdc = xy;
    gl_Position = vec4(xy, 0.0, 1.0);
}
"""

FRAGMENT = """#version 410 core
in vec2 vNdc;
out vec4 fragColor;

uniform sampler3D uVol;      // intensity, linear filter; texture axes (s, t, r) = (l, k, h)
uniform sampler3D uMask;     // 1 where a voxel is measured, linear filter
uniform sampler3D uBlock;    // largest value of each block and its neighbours, nearest filter
uniform sampler2D uCmap;     // colour map, 256 x 1
uniform vec3 uDims;          // voxels along s, t, r
uniform vec3 uCDims;         // uDims / block size
uniform ivec3 uCSize;        // size of the block texture
uniform mat4 uW2T;           // world to texture coordinates
uniform mat4 uT2H;           // texture coordinates to Miller indices (h, k, l)
uniform vec3 uCam, uRight, uUp, uFwd;
uniform float uAspect, uTan, uHalf;
uniform int uOrtho;
uniform vec3 uLo, uHi;       // shown box in texture coordinates
uniform vec3 uSLo, uSHi;     // centres of the first and the last voxel in the box: limits of a sample
uniform int uMode;           // 0 surfaces, 1 projection, 2 distance of the first hit, 3 largest value
uniform int uNIso;
uniform vec3 uIso;           // levels, lowest first
uniform vec4 uCol0, uCol1, uCol2;
uniform float uStep;         // step in voxels
uniform int uMaxSteps;
uniform int uSkip;
uniform float uJitter;
uniform vec2 uLogRange;      // natural logarithm of the limits of the colour scale
uniform int uGrid;
uniform vec3 uGridStep;
uniform vec3 uBg, uGridCol, uEdgeCol;
uniform vec3 uCenter;
uniform float uRadius;
uniform float uFloor;        // smallest intensity that has a logarithm
uniform float uTScale;       // modes 2 and 3: the colour holds distance / uTScale

// A point on a face of the box has the value of the voxel inside the box.
float vol(vec3 p) { return textureLod(uVol, clamp(p, uSLo, uSHi), 0.0).r; }
float lvol(vec3 p) { float v = vol(p); return log(v > uFloor ? v : uFloor); }

vec4 pack24(float x) {
    uint u = uint(clamp(x, 0.0, 1.0) * 16777215.0 + 0.5);
    return vec4(float((u >> 16) & 255u), float((u >> 8) & 255u), float(u & 255u), 255.0) / 255.0;
}

vec3 shade(vec3 p, vec3 pw, vec3 rd, vec3 base) {
    vec3 e = 1.0 / uDims;
    vec3 g = vec3(lvol(p + vec3(e.x, 0.0, 0.0)) - lvol(p - vec3(e.x, 0.0, 0.0)),
                  lvol(p + vec3(0.0, e.y, 0.0)) - lvol(p - vec3(0.0, e.y, 0.0)),
                  lvol(p + vec3(0.0, 0.0, e.z)) - lvol(p - vec3(0.0, 0.0, e.z))) * uDims;
    vec3 n = -normalize(transpose(mat3(uW2T)) * g + vec3(1e-20));
    if (dot(n, rd) > 0.0) n = -n;
    vec3 L = normalize(-rd + 0.45 * uUp - 0.25 * uRight);
    float diff = max(dot(n, L), 0.0);
    float spec = pow(max(dot(n, normalize(L - rd)), 0.0), 40.0);
    float rim = 1.0 - max(dot(n, -rd), 0.0);
    vec3 c = base * (0.32 + 0.68 * diff) * (1.0 - 0.30 * rim * rim) + 0.15 * spec;
    float depth = clamp(dot(pw - uCenter, uFwd) / uRadius, -1.0, 1.0) * 0.5 + 0.5;
    return mix(c, uBg, 0.22 * depth);
}

void main() {
    vec3 ro, rd;
    if (uOrtho == 1) {
        ro = uCam + (vNdc.x * uAspect * uRight + vNdc.y * uUp) * uHalf;
        rd = uFwd;
    } else {
        ro = uCam;
        rd = normalize(uFwd + (vNdc.x * uAspect * uRight + vNdc.y * uUp) * uTan);
    }
    vec3 o = (uW2T * vec4(ro, 1.0)).xyz;
    vec3 d = mat3(uW2T) * rd;
    d = mix(d, vec3(1e-9), lessThan(abs(d), vec3(1e-9)));
    vec3 inv = 1.0 / d;
    vec3 ta = (uLo - o) * inv, tb = (uHi - o) * inv;
    vec3 t0 = min(ta, tb), t1 = max(ta, tb);
    float tn = max(max(t0.x, t0.y), t0.z);
    float tf = min(min(t1.x, t1.y), t1.z);
    if (uOrtho == 0) tn = max(tn, 0.0);

    // far faces: grid of the Miller indices and edges of the box (derivatives before a branch)
    vec3 pFar = o + tf * d;
    vec3 pNear = o + tn * d;
    vec3 hFar = (uT2H * vec4(pFar, 1.0)).xyz / uGridStep;
    vec3 fwH = max(fwidth(hFar), vec3(1e-7));
    vec3 fwF = max(fwidth(pFar), vec3(1e-9));
    vec3 fwN = max(fwidth(pNear), vec3(1e-9));
    vec3 col = uBg;
    if (tf <= tn) { fragColor = (uMode >= 2) ? vec4(0.0) : vec4(col, 1.0); return; }

    if (uGrid == 1) {
        vec3 face = vec3(equal(t1, vec3(tf)));               // texture axis of the far face
        vec3 want = 1.0 - face.zyx;                          // the two indices that change on this face
        vec3 dist = abs(fract(hFar + 0.5) - 0.5) / fwH;      // distance to the next line in pixels
        vec3 dense = smoothstep(4.0, 10.0, 1.0 / fwH);       // no lines where they are closer than 4 pixels
        vec3 line = (1.0 - smoothstep(0.4, 1.2, dist)) * want * dense;
        col = mix(col, uGridCol, 0.45 * max(max(line.x, line.y), line.z));
    }
    vec3 de = min(pFar - uLo, uHi - pFar) / fwF;
    if (dot(vec3(lessThan(de, vec3(1.3))), vec3(1.0)) >= 2.0) col = mix(col, uGridCol, 0.95);

    float dt = uStep / length(d * uDims);
    ivec3 last = ivec3(-1);

    if (uMode == 1 || uMode == 3) {                          // largest value along the ray
        float noise = fract(52.9829189 * fract(dot(gl_FragCoord.xy, vec2(0.06711056, 0.00583715))));
        float t = tn + dt * noise * uJitter;
        float m = 0.0;
        for (int i = 0; i < uMaxSteps; i++) {
            if (t > tf) break;
            vec3 p = o + t * d;
            if (uSkip == 1) {
                ivec3 c = clamp(ivec3(floor(p * uCDims)), ivec3(0), uCSize - 1);
                if (any(notEqual(c, last))) {
                    last = c;
                    if (texelFetch(uBlock, c, 0).r <= m) {
                        vec3 tc = ((vec3(c) + step(vec3(0.0), d)) / uCDims - o) * inv;
                        t = max(min(min(tc.x, tc.y), tc.z), t) + 0.02 * dt;
                        continue;
                    }
                }
            }
            m = max(m, vol(p));
            t += dt;
        }
        float c = clamp((log(max(m, uFloor)) - uLogRange.x) / (uLogRange.y - uLogRange.x), 0.0, 1.0);
        if (uMode == 3) { fragColor = pack24(c); return; }
        col = mix(col, texture(uCmap, vec2(c, 0.5)).rgb, smoothstep(0.0, 0.22, c));
    } else {                                                 // surfaces
        vec4 acc = vec4(0.0);
        float v0 = vol(o + (tn + 1e-3 * dt) * d);
        bvec3 ins = bvec3(false);
        for (int k = 0; k < uNIso; k++) {                    // the box cuts a region above a level: flat cap
            if (v0 >= uIso[k]) {
                ins[k] = true;
                if (uMode == 2 && k == 0) { fragColor = pack24(tn / uTScale); return; }
                vec4 ck = (k == 0) ? uCol0 : ((k == 1) ? uCol1 : uCol2);
                acc.rgb += (1.0 - acc.a) * ck.a * mix(ck.rgb * 0.78, uBg, 0.10);
                acc.a += (1.0 - acc.a) * ck.a;
            }
        }
        float prev = v0, tPrev = tn;
        float t = tn + dt;
        for (int i = 0; i < uMaxSteps; i++) {
            if (t > tf || acc.a > 0.985) break;
            vec3 p = o + t * d;
            if (uSkip == 1 && !any(ins)) {
                ivec3 c = clamp(ivec3(floor(p * uCDims)), ivec3(0), uCSize - 1);
                if (any(notEqual(c, last))) {
                    last = c;
                    if (texelFetch(uBlock, c, 0).r < uIso.x) {
                        vec3 tc = ((vec3(c) + step(vec3(0.0), d)) / uCDims - o) * inv;
                        t = max(min(min(tc.x, tc.y), tc.z), t) + 0.02 * dt;
                        prev = 0.0; tPrev = t;
                        continue;
                    }
                }
            }
            float v = vol(p);
            for (int k = 0; k < uNIso; k++) {
                bool now = v >= uIso[k];
                if (now != ins[k]) {
                    float tA = tPrev, tB = t, vA = prev, vB = v;     // the level is between tA and tB
                    for (int j = 0; j < 4; j++) {
                        float tM = 0.5 * (tA + tB);
                        float vM = vol(o + tM * d);
                        if ((vM >= uIso[k]) == now) { tB = tM; vB = vM; } else { tA = tM; vA = vM; }
                    }
                    float a = log(max(vA, uFloor)), b = log(max(vB, uFloor));
                    float f = (b != a) ? clamp((log(uIso[k]) - a) / (b - a), 0.0, 1.0) : 1.0;
                    float th = mix(tA, tB, f);
                    if (uMode == 2 && k == 0) { fragColor = pack24(th / uTScale); return; }
                    vec3 ph = o + th * d;
                    vec4 ck = (k == 0) ? uCol0 : ((k == 1) ? uCol1 : uCol2);
                    vec3 base = (textureLod(uMask, clamp(ph, uSLo, uSHi), 0.0).r < 0.97) ? uEdgeCol : ck.rgb;
                    vec3 c3 = shade(ph, ro + th * rd, rd, base);
                    acc.rgb += (1.0 - acc.a) * ck.a * c3;
                    acc.a += (1.0 - acc.a) * ck.a;
                    ins[k] = now;
                }
            }
            prev = v; tPrev = t; t += dt;
        }
        if (uMode == 2) { fragColor = vec4(0.0); return; }
        col = acc.rgb + (1.0 - acc.a) * col;
    }

    vec3 dn = min(pNear - uLo, uHi - pNear) / fwN;           // edges of the near faces
    if (dot(vec3(lessThan(dn, vec3(1.0))), vec3(1.0)) >= 2.0) col = mix(col, uGridCol, 0.45);
    fragColor = vec4(col, 1.0);
}
"""

# Colours of the 3D image: background, grid lines, the grey of a surface at the edge of the
# data, and the labels (text: names of the axes, soft: numbers).
LIGHT = {'bg': (1.0, 1.0, 1.0), 'grid': (0.32, 0.37, 0.44), 'edge': (0.60, 0.62, 0.65),
         'text': '#1B2330', 'soft': '#525E6F'}
DARK = {'bg': (0.055, 0.065, 0.085), 'grid': (0.62, 0.66, 0.72), 'edge': (0.50, 0.52, 0.55),
        'text': '#E8ECF2', 'soft': '#B4BCC8'}


def gl_problem() -> str | None:
    """A reason why the 3D view cannot draw in this process, or None."""
    app = QGuiApplication.instance()
    if app is None:
        return 'There is no Qt application.'
    if app.platformName() in ('offscreen', 'minimal'):
        return f'The Qt platform "{app.platformName()}" has no OpenGL.'
    return None


def surface_format() -> QSurfaceFormat:
    fmt = QSurfaceFormat()
    fmt.setVersion(4, 1)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    return fmt


def gl_functions(ctx: QOpenGLContext) -> Any:
    """The OpenGL 4.1 functions of a context. RuntimeError if the context is older."""
    prof = QOpenGLVersionProfile()
    prof.setVersion(4, 1)
    prof.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    try:
        gl = QOpenGLVersionFunctionsFactory.get(prof, ctx)
    except Exception:                                   # PyQt6 raises for a version that it does not wrap
        gl = None
    if gl is None:
        f = ctx.format()
        raise RuntimeError('The 3D view needs OpenGL 4.1. This computer gives OpenGL '
                           f'{f.majorVersion()}.{f.minorVersion()}. Update the graphics driver.')
    return gl


def _vec3(v: Any) -> QVector3D:
    return QVector3D(float(v[0]), float(v[1]), float(v[2]))


def _mat4(m: Any) -> QMatrix4x4:
    return QMatrix4x4(*[float(x) for x in np.asarray(m, dtype=np.float64).reshape(16)])


def image_array(img: QImage) -> npt.NDArray[np.uint8]:
    """The pixels of an image as an array (rows, columns, 4) in the order R, G, B, A."""
    img = img.convertToFormat(QImage.Format.Format_RGBA8888)
    buf = img.constBits()
    buf.setsize(img.sizeInBytes())
    arr = np.frombuffer(buf, np.uint8).reshape(img.height(), img.bytesPerLine())
    return arr[:, :img.width() * 4].reshape(img.height(), img.width(), 4).copy()


def unpack24(rgba: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """The numbers 0..1 that the shader wrote in the test modes. NaN where it wrote nothing."""
    r, g, b = (rgba[..., i].astype(np.float64) for i in range(3))
    x = (r * 65536.0 + g * 256.0 + b) / 16777215.0
    return np.where(rgba[..., 3] > 0, x, np.nan)


class RayCaster:
    """The textures of one volume and the draw call. The caller makes the context current."""

    def __init__(self, gl: Any) -> None:
        self.gl = gl
        self.prog = QOpenGLShaderProgram()
        if not (self.prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, VERTEX)
                and self.prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, FRAGMENT)
                and self.prog.link()):
            raise RuntimeError('The shader of the 3D view did not compile:\n' + self.prog.log())
        self.vao = QOpenGLVertexArrayObject()
        self.vao.create()
        self.tex: QOpenGLTexture | None = None
        self.mask: QOpenGLTexture | None = None
        self.block: QOpenGLTexture | None = None
        self.cmap: QOpenGLTexture | None = None
        self.cmap_name = ''
        self.grid: v3.Grid | None = None
        self.timing: dict[str, float] = {}

    @property
    def ready(self) -> bool:
        return self.tex is not None

    def _texture3d(self, size: tuple[int, int, int], fmt: Any, pixel_type: Any, linear: bool) -> QOpenGLTexture:
        tex = QOpenGLTexture(QOpenGLTexture.Target.Target3D)
        tex.setSize(*size)
        tex.setFormat(fmt)
        tex.setMipLevels(1)
        tex.allocateStorage(QOpenGLTexture.PixelFormat.Red, pixel_type)
        f = QOpenGLTexture.Filter.Linear if linear else QOpenGLTexture.Filter.Nearest
        tex.setMinMagFilters(f, f)
        tex.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        return tex

    def load(self, data: npt.NDArray[Any], grid: v3.Grid) -> None:
        """Send a volume to the GPU. `data` has the shape `grid.shape`. MemoryError if it does not fit."""
        self.release()
        gl = self.gl
        while gl.glGetError():                           # forget earlier errors
            pass
        nh, nk, nl = data.shape
        # Rows of bytes with a length that is not a multiple of 4 need an alignment of 1.
        # PyQt6 6.10 does not apply a QOpenGLPixelTransferOptions object (the upload then
        # reads past the end of the array), so the alignment is set in OpenGL itself.
        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        red, f32, u8 = (QOpenGLTexture.PixelFormat.Red, QOpenGLTexture.PixelType.Float32,
                        QOpenGLTexture.PixelType.UInt8)
        t = time.perf_counter()
        tex = self._texture3d((nl, nk, nh), QOpenGLTexture.TextureFormat.R32F, f32, True)
        mask = self._texture3d((nl, nk, nh), QOpenGLTexture.TextureFormat.R8_UNorm, u8, True)
        slab = max(1, min(32, (16 * 2**20) // max(nk * nl, 1)))
        for z in range(0, nh, slab):
            part = np.asarray(data[z:z + slab], dtype=np.float32)
            ok = v3.measured(part)
            part = np.ascontiguousarray(np.nan_to_num(part, nan=0.0, posinf=3e38, neginf=0.0))
            tex.setData(0, 0, z, nl, nk, part.shape[0], red, f32, part)
            mask.setData(0, 0, z, nl, nk, part.shape[0], red, u8,
                         np.ascontiguousarray(ok.astype(np.uint8) * 255))
        gl.glFinish()
        err = gl.glGetError()
        if err:
            tex.destroy()
            mask.destroy()
            if err == GL_OUT_OF_MEMORY:
                raise MemoryError('The volume does not fit in the memory of the graphics card.')
            raise RuntimeError(f'OpenGL error {err:#06x} when the volume was sent to the graphics card.')
        self.timing['upload_s'] = time.perf_counter() - t
        t = time.perf_counter()
        bm = v3.block_max(data)
        self.timing['block_max_s'] = time.perf_counter() - t
        block = self._texture3d((bm.shape[2], bm.shape[1], bm.shape[0]),
                                QOpenGLTexture.TextureFormat.R32F, f32, False)
        block.setData(red, f32, np.ascontiguousarray(bm))
        gl.glPixelStorei(GL_UNPACK_ALIGNMENT, 4)
        self.tex, self.mask, self.block, self.grid = tex, mask, block, grid
        self.dims = (nl, nk, nh)
        self.cdims = (bm.shape[2], bm.shape[1], bm.shape[0])

    def set_colormap(self, name: str) -> None:
        if name == self.cmap_name and self.cmap is not None:
            return
        from matplotlib import colormaps
        if name not in colormaps:                        # a name that matplotlib does not have
            name = 'inferno_r' if name.endswith('_r') else 'inferno'
            if name == self.cmap_name and self.cmap is not None:
                return
        lut = np.ascontiguousarray((colormaps[name](np.linspace(0.0, 1.0, 256)) * 255).astype(np.uint8))
        if self.cmap is not None:
            self.cmap.destroy()
        cm = QOpenGLTexture(QOpenGLTexture.Target.Target2D)
        cm.setSize(256, 1)
        cm.setFormat(QOpenGLTexture.TextureFormat.RGBA8_UNorm)
        cm.setMipLevels(1)
        cm.allocateStorage(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8)
        cm.setMinMagFilters(QOpenGLTexture.Filter.Linear, QOpenGLTexture.Filter.Linear)
        cm.setWrapMode(QOpenGLTexture.WrapMode.ClampToEdge)
        cm.setData(QOpenGLTexture.PixelFormat.RGBA, QOpenGLTexture.PixelType.UInt8, lut)
        self.cmap, self.cmap_name = cm, name

    def release(self) -> None:
        """Free the textures of the volume."""
        for name in ('tex', 'mask', 'block'):
            tex = getattr(self, name)
            if tex is not None:
                tex.destroy()
                setattr(self, name, None)

    def destroy(self) -> None:
        """Free all OpenGL objects. The context must be current. The object cannot draw after it."""
        self.release()
        if self.cmap is not None:
            self.cmap.destroy()
            self.cmap, self.cmap_name = None, ''
        self.vao.destroy()
        self.prog.removeAllShaders()

    def draw(self, width: int, height: int, scene: v3.Scene, cam: v3.Camera) -> None:
        """Draw the scene into the framebuffer that is bound now."""
        assert self.tex is not None and self.mask is not None and self.block is not None
        assert self.grid is not None
        gl, p, grid = self.gl, self.prog, self.grid
        self.set_colormap(colormap_for(scene))
        assert self.cmap is not None
        gl.glViewport(0, 0, width, height)
        p.bind()
        self.tex.bind(0)
        self.mask.bind(1)
        self.block.bind(2)
        self.cmap.bind(3)
        for i, name in enumerate(('uVol', 'uMask', 'uBlock', 'uCmap')):
            p.setUniformValue(name, i)
        p.setUniformValue('uDims', _vec3(self.dims))
        p.setUniformValue('uCDims', _vec3(np.array(self.dims) / v3.BLOCK))
        gl.glUniform3i(p.uniformLocation('uCSize'), *self.cdims)
        p.setUniformValue('uW2T', _mat4(grid.w2t))
        p.setUniformValue('uT2H', _mat4(grid.t2h))
        fwd, right, up = cam.basis()
        p.setUniformValue('uCam', _vec3(cam.eye(grid)))
        p.setUniformValue('uRight', _vec3(right))
        p.setUniformValue('uUp', _vec3(up))
        p.setUniformValue('uFwd', _vec3(fwd))
        p.setUniformValue('uAspect', float(width) / float(height))
        p.setUniformValue('uTan', float(v3.TAN_HALF_FOV))
        p.setUniformValue('uHalf', float(cam.half_height(grid)))
        p.setUniformValue('uOrtho', 1 if cam.ortho else 0)
        p.setUniformValue('uLo', _vec3(scene.lo))
        p.setUniformValue('uHi', _vec3(scene.hi))
        s_lo, s_hi = v3.sample_limits(grid.shape, scene.lo, scene.hi)
        p.setUniformValue('uSLo', _vec3(s_lo))
        p.setUniformValue('uSHi', _vec3(s_hi))
        mode = {'surface': 0, 'projection': 1}[scene.mode]
        if scene.debug:
            mode = 2 if scene.mode == 'surface' else 3
        p.setUniformValue('uMode', mode)
        levels = scene.sorted_levels()
        p.setUniformValue('uNIso', len(levels))
        p.setUniformValue('uIso', _vec3(([lv.value for lv in levels] + [3e38] * 3)[:3]))
        for i in range(3):
            lv = levels[i] if i < len(levels) else v3.Level(1.0)
            p.setUniformValue(f'uCol{i}', QVector4D(float(lv.color[0]), float(lv.color[1]),
                                                    float(lv.color[2]), float(lv.opacity)))
        p.setUniformValue('uStep', float(scene.step))
        p.setUniformValue('uMaxSteps', int(2.0 * float(np.linalg.norm(self.dims)) / scene.step) + 64)
        p.setUniformValue('uSkip', 1 if scene.skip else 0)
        p.setUniformValue('uJitter', float(scene.jitter))
        lo, hi = max(scene.clim[0], scene.floor), max(scene.clim[1], scene.floor * 1.0001)
        if hi <= lo:
            hi = lo * 1.0001
        p.setUniformValue('uLogRange', QVector2D(math.log(lo), math.log(hi)))
        p.setUniformValue('uGrid', 1 if scene.grid else 0)
        p.setUniformValue('uGridStep', _vec3(scene.grid_step))
        theme = DARK if scene.dark else LIGHT
        p.setUniformValue('uBg', _vec3(theme['bg']))
        p.setUniformValue('uGridCol', _vec3(theme['grid']))
        p.setUniformValue('uEdgeCol', _vec3(theme['edge']))
        p.setUniformValue('uCenter', _vec3(grid.center))
        p.setUniformValue('uRadius', float(grid.radius))
        p.setUniformValue('uFloor', float(scene.floor))
        p.setUniformValue('uTScale', float(t_scale(grid, cam)))
        self.vao.bind()
        gl.glDrawArrays(GL_TRIANGLES, 0, 3)
        self.vao.release()
        p.release()
        gl.glActiveTexture(0x84C0)                       # texture unit 0 for Qt


def colormap_for(scene: v3.Scene) -> str:
    """Name of the matplotlib colour map that is drawn.

    The weak end of the map must be near the background. A map goes from dark
    to light, so it is used as it is on a dark background and reversed on a
    light background. A name that ends with "_r" is a map from light to dark.
    """
    name = scene.colormap
    light_first = name.endswith('_r')
    if light_first == scene.dark:                        # the weak end is not at the background
        return name[:-2] if light_first else name + '_r'
    return name


def t_scale(grid: v3.Grid, cam: v3.Camera) -> float:
    """Longest distance along a ray that the test modes can store."""
    return 16.0 * grid.radius + 4.0 * float(np.linalg.norm(cam.target)) + cam.half_height(grid) / v3.TAN_HALF_FOV


class Offscreen:
    """Draws images of a volume without a window (tests, scripts)."""

    def __init__(self) -> None:
        problem = gl_problem()
        if problem:
            raise RuntimeError(problem)
        self.ctx = QOpenGLContext()
        self.ctx.setFormat(surface_format())
        if not self.ctx.create():
            raise RuntimeError('This computer did not give an OpenGL context.')
        self.surface = QOffscreenSurface()
        self.surface.setFormat(self.ctx.format())
        self.surface.create()
        if not self.ctx.makeCurrent(self.surface):
            raise RuntimeError('The OpenGL context cannot be used.')
        self.gl = gl_functions(self.ctx)
        self.rc = RayCaster(self.gl)

    def info(self) -> dict[str, Any]:
        return {'renderer': self.gl.glGetString(GL_RENDERER), 'version': self.gl.glGetString(GL_VERSION),
                'max_3d_texture': int(self.gl.glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE))}

    def load(self, data: npt.NDArray[Any], grid: v3.Grid) -> None:
        self.ctx.makeCurrent(self.surface)
        self.rc.load(data, grid)

    def render(self, scene: v3.Scene, cam: v3.Camera, width: int, height: int) -> QImage:
        self.ctx.makeCurrent(self.surface)
        fbo = QOpenGLFramebufferObject(width, height)
        fbo.bind()
        self.rc.draw(width, height, scene, cam)
        self.gl.glFinish()
        img = fbo.toImage()
        fbo.release()
        return img

    def frame_ms(self, scene: v3.Scene, cam: v3.Camera, width: int, height: int, frames: int = 12) -> float:
        """Median time of a frame in milliseconds. The camera turns a little for each frame."""
        self.ctx.makeCurrent(self.surface)
        fbo = QOpenGLFramebufferObject(width, height)
        fbo.bind()
        times = []
        yaw = cam.yaw
        for i in range(frames):
            cam.yaw = yaw + 0.05 * i
            t = time.perf_counter()
            self.rc.draw(width, height, scene, cam)
            self.gl.glFinish()
            times.append(time.perf_counter() - t)
        cam.yaw = yaw
        fbo.release()
        return 1.0e3 * float(np.median(times))

    def close(self) -> None:
        self.ctx.makeCurrent(self.surface)
        self.rc.destroy()
        self.ctx.doneCurrent()


class View3DCanvas(QOpenGLWidget):
    """The widget that shows the volume. The window sets `scene` and `camera` and calls `update()`."""

    failed = pyqtSignal(str)             # the view cannot draw (the text says why)
    loaded = pyqtSignal()                # the volume is on the graphics card
    view_changed = pyqtSignal()          # the camera moved
    hover = pyqtSignal(object)           # the point below the mouse (dict) or None
    frame_done = pyqtSignal(float, float)    # milliseconds of the frame, part of the pixels (0..1)

    def __init__(self, parent: Any = None) -> None:
        super().__init__(parent)
        self.setFormat(surface_format())
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)
        self.setMinimumSize(320, 240)
        self.scene = v3.Scene()
        self.camera = v3.Camera()
        self.grid: v3.Grid | None = None
        self.data: npt.NDArray[Any] | None = None
        self.error = ''
        self.info: dict[str, Any] = {}
        self.overlay: Any = None             # a widget over the canvas that is drawn again with it
        self._gl: Any = None
        self._rc: RayCaster | None = None
        self._need_upload = False
        self._low: QOpenGLFramebufferObject | None = None
        self._moving = False
        self._scale = 1.0
        self._tune = False
        self._press: QPointF | None = None
        self._mouse: QPointF | None = None
        self._idle = QTimer(self)
        self._idle.setSingleShot(True)
        self._idle.setInterval(160)
        self._idle.timeout.connect(self._stop_moving)
        self._pick = QTimer(self)
        self._pick.setSingleShot(True)
        self._pick.setInterval(40)
        self._pick.timeout.connect(self._emit_hover)

    # ---- data

    def set_data(self, data: npt.NDArray[Any], grid: v3.Grid) -> None:
        """Show a volume. `data` is the array of the texture (shape `grid.shape`)."""
        self.data, self.grid = data, grid
        self._need_upload = True
        self._tune = True
        self.refresh()

    def refresh(self) -> None:
        """Draw the scene again, and the labels over it."""
        self.update()
        if self.overlay is not None:
            self.overlay.update()

    def retune(self) -> None:
        """Measure again whether the jump over empty blocks makes the frames faster."""
        self._tune = True

    def touch(self) -> None:
        """Tell the canvas that the user changes the scene step by step (a drag of a level).

        Slow graphics then draw with fewer pixels until the changes stop.
        """
        self._start_moving()

    # ---- OpenGL

    def initializeGL(self) -> None:
        try:
            self._gl = gl_functions(self.context())
            self._rc = RayCaster(self._gl)
        except RuntimeError as e:
            self._fail(str(e))
            return
        self.info = {'renderer': self._gl.glGetString(GL_RENDERER),
                     'version': self._gl.glGetString(GL_VERSION),
                     'max_3d_texture': int(self._gl.glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE))}
        self.context().aboutToBeDestroyed.connect(self._cleanup)
        # Qt makes a new context when the widget goes to another window: the textures are gone then
        self._need_upload = self.data is not None
        self._low = None
        self._tune = True

    def _fail(self, text: str) -> None:
        self.error = text
        self._rc = None
        self.failed.emit(text)

    def _cleanup(self) -> None:
        if self._rc is not None:
            self.makeCurrent()
            self._rc.destroy()
            self._low = None
            self.doneCurrent()
            self._rc = None

    def _draw(self, width: int, height: int, scale: float) -> float:
        """One frame into the widget. Returns the time in milliseconds."""
        assert self._rc is not None
        gl = self._gl
        t = time.perf_counter()
        target = self.defaultFramebufferObject()
        if scale < 0.999:
            w, h = max(int(width * scale), 1), max(int(height * scale), 1)
            if self._low is None or self._low.width() != w or self._low.height() != h:
                self._low = QOpenGLFramebufferObject(w, h)
            self._low.bind()
            self._rc.draw(w, h, self.scene, self.camera)
            gl.glBindFramebuffer(GL_READ_FRAMEBUFFER, self._low.handle())
            gl.glBindFramebuffer(GL_DRAW_FRAMEBUFFER, target)
            gl.glBlitFramebuffer(0, 0, w, h, 0, 0, width, height, GL_COLOR_BUFFER_BIT, GL_LINEAR)
            gl.glBindFramebuffer(GL_FRAMEBUFFER, target)
        else:
            gl.glBindFramebuffer(GL_FRAMEBUFFER, target)
            self._rc.draw(width, height, self.scene, self.camera)
        gl.glFinish()
        return 1.0e3 * (time.perf_counter() - t)

    def paintGL(self) -> None:
        try:
            self._paint()
        except Exception as e:                           # an error must not leave paintGL: Qt would stop the program
            self._fail(f'{type(e).__name__}: {e}')

    def _paint(self) -> None:
        gl = self._gl
        if gl is None or self._rc is None:
            return
        if self._need_upload and self.data is not None and self.grid is not None:
            self._need_upload = False
            self._rc.load(self.data, self.grid)
            self.loaded.emit()
        size = self.size() * self.devicePixelRatioF()    # Qt rounds as for its own framebuffer
        width, height = max(size.width(), 1), max(size.height(), 1)
        if not self._rc.ready:
            bg = (DARK if self.scene.dark else LIGHT)['bg']
            gl.glClearColor(bg[0], bg[1], bg[2], 1.0)
            gl.glClear(GL_COLOR_BUFFER_BIT)
            return
        layout = v3.axis_layout(self.grid, self.camera, self.scene.lo, self.scene.hi,
                                max(self.width(), 1), max(self.height(), 1))
        self.scene.grid_step = v3.grid_steps(layout)     # the lines are where the labels are
        if self._tune and not self._moving:              # which is faster on this graphics card?
            self._tune = False
            times = {}
            for skip in (True, False):                   # with a quarter of the pixels: it is a test
                self.scene.skip = skip
                times[skip] = min(self._draw(width, height, 0.5) for _ in range(2))
            self.scene.skip = times[True] <= times[False]
        scale = self._scale if self._moving else 1.0
        ms = self._draw(width, height, scale)
        full = ms / (scale * scale)                      # time of this frame with all pixels
        if full > FRAME_MS * 1.2:
            want = math.sqrt(FRAME_MS / full)            # a few fixed sizes: the small framebuffer is kept
            self._scale = next((s for s in SCALES if s <= want), SCALES[-1])
        elif full < FRAME_MS * 0.6:
            self._scale = 1.0
        self.frame_done.emit(ms, scale)

    def grab_image(self, width: int, height: int) -> QImage | None:
        """The scene as an image of the given size in pixels (without the labels)."""
        if self._rc is None or not self._rc.ready:
            return None
        self.makeCurrent()
        fbo = QOpenGLFramebufferObject(width, height)
        fbo.bind()
        self._rc.draw(width, height, self.scene, self.camera)
        self._gl.glFinish()
        img = fbo.toImage()
        fbo.release()
        self.doneCurrent()
        return img

    # ---- mouse

    def _start_moving(self) -> None:
        self._moving = True
        self._idle.start()

    def _stop_moving(self) -> None:
        if self._press is not None:                      # a button is still down
            self._idle.start()
            return
        self._moving = False
        self.refresh()

    def mousePressEvent(self, event: Any) -> None:
        self._press = event.position()
        self._start_moving()

    def mouseReleaseEvent(self, event: Any) -> None:
        self._press = None
        self._idle.start()

    def mouseMoveEvent(self, event: Any) -> None:
        pos = event.position()
        self._mouse = pos
        if self._press is None or self.grid is None:
            self._pick.start()
            return
        dx, dy = pos.x() - self._press.x(), pos.y() - self._press.y()
        self._press = pos
        buttons, mods = event.buttons(), event.modifiers()
        move = (buttons & (Qt.MouseButton.RightButton | Qt.MouseButton.MiddleButton)
                or mods & Qt.KeyboardModifier.ShiftModifier)
        if move:
            self.camera.pan(self.grid, dx, dy, max(self.height(), 1))
        else:
            self.camera.orbit(dx, dy)
        self._start_moving()
        self.view_changed.emit()
        self.refresh()

    def wheelEvent(self, event: Any) -> None:
        if self.grid is None:
            return
        pos = event.position()
        self.camera.zoom_at(self.grid, 2.0 ** (event.angleDelta().y() / 480.0), pos.x(), pos.y(),
                            max(self.width(), 1), max(self.height(), 1))
        self._start_moving()
        self.view_changed.emit()
        self.refresh()
        event.accept()

    def leaveEvent(self, event: Any) -> None:
        self._mouse = None
        self._pick.stop()
        self.hover.emit(None)

    def point_at(self, x: float, y: float) -> dict[str, Any] | None:
        """The point of the scene below the pixel (x, y) of the widget, or None."""
        if self.data is None or self.grid is None:
            return None
        origin, direction = self.camera.ray(self.grid, x, y, max(self.width(), 1), max(self.height(), 1))
        s = self.scene
        if s.mode == 'projection':
            return v3.ray_max(self.data, self.grid, origin, direction, s.lo, s.hi, s.step, self.camera.ortho)
        levels = s.sorted_levels()
        if not levels:
            return None
        return v3.first_hit(self.data, self.grid, origin, direction, levels[0].value, s.lo, s.hi,
                            s.step, s.floor, self.camera.ortho)

    def _emit_hover(self) -> None:
        if self._mouse is None or self._press is not None:
            return
        self.hover.emit(self.point_at(self._mouse.x(), self._mouse.y()))
