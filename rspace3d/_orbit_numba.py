"""
_orbit_numba.py: numba implementation of the orbit engine (optional).

The package does not need numba. `orbit_engine.run_cpu` uses this module
when numba is installed. Each CPU thread processes one block of voxels at a
time. The per-voxel code is a copy of the CUDA kernel in `orbit_engine`. It
uses the same operations in the same order, so the intensity, variance and
outlier count are bit-identical to the GPU and the NumPy engines. numba does
not fuse a multiply and an add unless `fastmath` is on. This module does not
use `fastmath`. Each float32 result is converted with np.float32(), so
that it is rounded to float32 also if numba calculates it in float64.

The compiled code is kept in the numba cache (`cache=True`). Thus only the
first call after an installation or a change of this file compiles it
(about 10 s). A frozen application (for example a PyInstaller build) does
not use the cache. It compiles the code once per run.

`var_mode` selects the member variance: 0 no variance, 1 the stored
variance `var`, 2 the derived variance max(x, 1) of the data value x.
"""

from __future__ import annotations

import sys

import numpy as np
from numba import njit, prange

NSH = 10        # |q| shells, the same as orbit_engine.N_SHELLS
NI = 6          # integer sums per shell
NF = 4          # float sums per shell
I_OUT = 5       # column of n_outliers_removed in the integer sums
MAXOPS = 48
# A frozen application has no writable source folder for the numba cache
_CACHE = not getattr(sys, 'frozen', False)


@njit(nogil=True, cache=_CACHE)
def _isort(a, n):
    """Insertion sort of a[:n] (the same order of operations as the CUDA kernel)."""
    for i in range(1, n):
        x = a[i]
        j = i
        while j > 0 and a[j - 1] > x:
            a[j] = a[j - 1]
            j -= 1
        a[j] = x


@njit(nogil=True, cache=_CACHE)
def _med_sorted(a, n):
    h = n >> 1
    if n & 1:
        return a[h]
    return np.float32((a[h - 1] + a[h]) * np.float32(0.5))


@njit(parallel=True, nogil=True, cache=_CACHE)
def orbit_blocks(vol, var, var_mode, out, out_var, tab, opi, opl, opd, geo,
                 nh, nk, nl, n_ops, id_op, do_reject, sigma, min_valid, floor_on,
                 edge_tol, mad_scale, den_min, base, count, block, iacc, facc):
    """Process the voxels base .. base + count - 1.

    `out` and `out_var` are indexed from `base`. Block b adds its diagnostic
    sums to iacc[b] (int64, (NSH, NI)) and facc[b] (float64, (NSH, NF)).
    """
    nblocks = (count + block - 1) // block
    nkl = nk * nl
    one = np.float32(1.0)
    zero32 = np.float32(0.0)
    nan32 = np.float32(np.nan)
    for b in prange(nblocks):
        vals = np.empty(MAXOPS, np.float32)
        mvar = np.empty(MAXOPS, np.float32)
        key = np.empty(MAXOPS, np.int64)
        srt = np.empty(MAXOPS, np.float32)
        surv = np.empty(MAXOPS, np.bool_)
        rep = np.empty(MAXOPS, np.int64)
        cnt = np.empty(MAXOPS, np.int64)
        seen = np.empty(MAXOPS, np.bool_)
        cc = np.empty(3, np.int64)
        nn = np.empty(3, np.int64)
        src = np.empty(3, np.int64)
        ib = np.empty(3, np.int64)
        w0 = np.empty(3, np.float64)
        w1 = np.empty(3, np.float64)
        hv = np.empty(3, np.float64)
        gv = np.empty(3, np.float64)
        nn[0] = nh
        nn[1] = nk
        nn[2] = nl
        li1 = min(count, (b + 1) * block)
        for li in range(b * block, li1):
            v = base + li
            c0 = v // nkl
            r = v - c0 * nkl
            c1 = r // nl
            c2 = r - c1 * nl
            cc[0] = c0
            cc[1] = c1
            cc[2] = c2
            dc0 = np.float64(c0)
            dc1 = np.float64(c1)
            dc2 = np.float64(c2)

            n = 0
            self_i = -1
            for o in range(n_ops):
                kind = opi[o, 0]
                x = zero32
                xv = zero32
                k = np.int64(0)
                if kind != 2:
                    ok = True
                    for d in range(3):
                        if kind == 0:
                            sd = tab[opi[o, 4 + d] + cc[opi[o, 1 + d]]]
                        else:
                            sd = (opl[o, 3 * d] * c0 + opl[o, 3 * d + 1] * c1
                                  + opl[o, 3 * d + 2] * c2 + opl[o, 9 + d])
                        if sd < 0 or sd >= nn[d]:
                            ok = False
                            break
                        src[d] = sd
                    if not ok:
                        continue
                    k = (src[0] * nk + src[1]) * nl + src[2]
                    x = vol[k]
                    if not np.isfinite(x):
                        continue
                    if var_mode == 1:
                        xv = var[k]
                    elif var_mode == 2:
                        xv = x if x > one else one
                else:
                    inside = True
                    for d in range(3):
                        c = (((opd[o, 3 * d] * dc0 + opd[o, 3 * d + 1] * dc1)
                              + opd[o, 3 * d + 2] * dc2) + opd[o, 9 + d])
                        hi = np.float64(nn[d] - 1)
                        if c < 0.0 and c > -edge_tol:
                            c = 0.0
                        if c > hi and c < hi + edge_tol:
                            c = hi
                        if not (c >= 0.0 and c <= hi):
                            inside = False
                            break
                        f = np.floor(c)
                        a0 = 1.0 - (c - f)
                        w0[d] = a0
                        w1[d] = 1.0 - a0
                        ib[d] = np.int64(f)
                    if not inside:
                        continue
                    num = 0.0
                    den = 0.0
                    for s0 in range(2):
                        a0 = w1[0] if s0 else w0[0]
                        if a0 == 0.0:
                            continue
                        for s1 in range(2):
                            a1 = w1[1] if s1 else w0[1]
                            if a1 == 0.0:
                                continue
                            for s2 in range(2):
                                a2 = w1[2] if s2 else w0[2]
                                if a2 == 0.0:
                                    continue
                                idx = ((ib[0] + s0) * nk + (ib[1] + s1)) * nl + ib[2] + s2
                                xx = vol[idx]
                                if not np.isfinite(xx):
                                    continue
                                num = num + ((np.float64(xx) * a0) * a1) * a2
                                den = den + (a0 * a1) * a2
                    num32 = np.float32(num)
                    den32 = np.float32(den)
                    if not (den32 > den_min):
                        continue
                    x = np.float32(num32 / (den32 if den32 > den_min else den_min))
                    if not np.isfinite(x):
                        continue
                    if var_mode != 0:
                        vs = 0.0
                        for s0 in range(2):
                            a0 = w1[0] if s0 else w0[0]
                            if a0 == 0.0:
                                continue
                            for s1 in range(2):
                                a1 = w1[1] if s1 else w0[1]
                                if a1 == 0.0:
                                    continue
                                for s2 in range(2):
                                    a2 = w1[2] if s2 else w0[2]
                                    if a2 == 0.0:
                                        continue
                                    idx = ((ib[0] + s0) * nk + (ib[1] + s1)) * nl + ib[2] + s2
                                    xx = vol[idx]
                                    if not np.isfinite(xx):
                                        continue
                                    rr = ((a0 * a1) * a2) / den
                                    if var_mode == 1:
                                        vk = np.float64(var[idx])
                                    else:
                                        vk = np.float64(xx if xx > one else one)
                                    vs = vs + (rr * rr) * vk
                        xv = np.float32(vs)
                    if w1[0] == 0.0 and w1[1] == 0.0 and w1[2] == 0.0:
                        k = (ib[0] * nk + ib[1]) * nl + ib[2]
                    else:
                        k = -1 - np.int64(o)
                if o == id_op:
                    self_i = n
                vals[n] = x
                mvar[n] = xv
                key[n] = k
                n += 1

            # median, MAD and rejection
            med = zero32
            thr = zero32
            tested = False
            floor_used = False
            if do_reject and n >= min_valid and n > 0:
                for i in range(n):
                    srt[i] = vals[i]
                _isort(srt, n)
                med = _med_sorted(srt, n)
                for i in range(n):
                    srt[i] = np.float32(abs(np.float32(vals[i] - med)))
                _isort(srt, n)
                scale = np.float32(_med_sorted(srt, n) * mad_scale)
                if floor_on:
                    root = np.float32(np.sqrt(med if med > one else one))
                    floor_used = root > scale
                    if root > scale:
                        scale = root
                thr = np.float32(sigma * scale)
                tested = True
            acc = zero32
            m = 0
            n_rej = 0
            self_rej = False
            for i in range(n):
                if tested and np.float32(abs(np.float32(vals[i] - med))) > thr:
                    surv[i] = False
                    n_rej += 1
                    if i == self_i:
                        self_rej = True
                else:
                    surv[i] = True
                    acc = np.float32(acc + vals[i])
                    m += 1
            res = np.float32(acc / np.float32(m)) if m > 0 else nan32
            out[li] = res

            # rep[i]: the first finite member that reads the same source as member i
            n_src = 0
            for i in range(n):
                r0 = i
                for j in range(i):
                    if key[j] == key[i]:
                        r0 = j
                        break
                rep[i] = r0
                if r0 == i:
                    n_src += 1

            chi = 0.0
            nd = 0
            if var_mode != 0:
                # cnt[g]: number of survivors that read the source of group g
                for i in range(n):
                    cnt[i] = 0
                    seen[i] = False
                for i in range(n):
                    if surv[i]:
                        cnt[rep[i]] += 1
                accv = 0.0
                mm = np.float64(m) * np.float64(m)
                mean = np.float64(res)
                for i in range(n):
                    if not surv[i]:
                        continue
                    g = rep[i]
                    first = not seen[g]
                    seen[g] = True
                    fct = np.float64(cnt[g]) / mm
                    accv = accv + fct * np.float64(mvar[i])
                    if first and mvar[i] > zero32:
                        dd = np.float64(vals[i]) - mean
                        chi = chi + (dd * dd) / np.float64(mvar[i])
                        nd += 1
                out_var[li] = np.float32(accv) if m > 0 else nan32

            if n > 0:
                for d in range(3):
                    hv[d] = geo[d] + ((geo[3 + 3 * d] * dc0 + geo[4 + 3 * d] * dc1)
                                      + geo[5 + 3 * d] * dc2)
                for d in range(3):
                    gv[d] = ((geo[12 + 3 * d] * hv[0] + geo[13 + 3 * d] * hv[1])
                             + geo[14 + 3 * d] * hv[2])
                q2 = (hv[0] * gv[0] + hv[1] * gv[1]) + hv[2] * gv[2]
                q = np.sqrt(q2 if q2 > 0.0 else 0.0)
                sh = np.int64(q * geo[21])
                if sh > NSH - 1:
                    sh = NSH - 1
                if sh < 0:
                    sh = 0
                iacc[b, sh, 0] += 1
                iacc[b, sh, 1] += n
                iacc[b, sh, 2] += n_rej
                if tested:
                    iacc[b, sh, 3] += 1
                if floor_used:
                    iacc[b, sh, 4] += 1
                if self_rej:
                    iacc[b, sh, I_OUT] += 1
                # the float sums have the weight 1 / (number of distinct sources)
                ds = np.float64(n_src)
                if var_mode != 0 and nd >= 2:
                    facc[b, sh, 0] += chi / ds
                    facc[b, sh, 1] += np.float64(nd - 1) / ds
                if m > 0:
                    mean = np.float64(res)
                    rn = 0.0
                    rd = 0.0
                    for i in range(n):
                        if not surv[i]:
                            continue
                        rn += abs(np.float64(vals[i]) - mean)
                        rd += abs(np.float64(vals[i]))
                    facc[b, sh, 2] += rn / ds
                    facc[b, sh, 3] += rd / ds
