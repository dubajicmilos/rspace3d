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

`wmode` selects the weights: 0 equal, 1 counts, 2 adaptive. `rmode` selects
the rejection: 0 none, 1 MAD, 2 noise model. `orbit_engine` explains them.
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


@njit(nogil=True, cache=_CACHE)
def _div(a, b):
    """a / b with the IEEE result for b = +0 (inf or NaN, as in CUDA and NumPy).

    The default numba error model raises an exception for a division by
    zero. (error_model='numpy' makes the engine about 10 % slower.) The
    divisors here are never -0.
    """
    if b != 0.0:
        return a / b
    if a > 0.0:
        return np.inf
    if a < 0.0:
        return -np.inf
    return np.nan


@njit(parallel=True, nogil=True, cache=_CACHE)
def orbit_blocks_basic(vol, var, var_mode, out, out_var, tab, opi, opl, opd, geo,
                       nh, nk, nl, n_ops, id_op, do_reject, sigma, min_valid, floor_on,
                       edge_tol, mad_scale, den_min, base, count, block, iacc, facc):
    """`orbit_blocks` for the options of rspace3d 2.3.0 only (equal weights, MAD or none).

    This is the code of rspace3d 2.3.0. It has no code for the other
    options. On a real volume of 2162 x 2068 x 61 voxels it used 5 % less
    time than `orbit_blocks`.
    `do_reject` True gives the MAD rejection. The result is bit-identical
    to `orbit_blocks` with wmode 0 and rmode 1 (`do_reject` True) or 0.
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


@njit(parallel=True, nogil=True, cache=_CACHE)
def orbit_blocks(vol, var, var_mode, out, out_var, tab, opi, opl, opd, geo,
                 nh, nk, nl, n_ops, id_op, rmode, sigma, min_valid, floor_on,
                 edge_tol, mad_scale, den_min, base, count, block, iacc, facc,
                 cnt, cnt_on, grd, g_on, wmode, gain, rel, kappa, flr, sigma_d):
    """Process the voxels base .. base + count - 1.

    `out` and `out_var` are indexed from `base`. Block b adds its diagnostic
    sums to iacc[b] (int64, (NSH, NI)) and facc[b] (float64, (NSH, NF)).
    `cnt` (pixel counts) and `grd` (gradient magnitude) are read only if
    `cnt_on` and `g_on` are true. Else N = 1 and G = 0.
    """
    need_n = wmode != 0 or rmode == 2
    read_n = need_n and cnt_on
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
        cnt_g = np.empty(MAXOPS, np.int64)
        seen = np.empty(MAXOPS, np.bool_)
        ncnt = np.empty(MAXOPS, np.float64)
        wg = np.empty(MAXOPS, np.float64)
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
                xn = 1.0
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
                    if read_n:
                        xn = np.float64(cnt[k])
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
                    numn = 0.0
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
                                if read_n:
                                    numn = numn + ((np.float64(cnt[idx]) * a0) * a1) * a2
                    num32 = np.float32(num)
                    den32 = np.float32(den)
                    if not (den32 > den_min):
                        continue
                    if read_n:
                        xn = numn / den
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
                if need_n:
                    ncnt[n] = xn
                n += 1

            # median, MAD and rejection
            med = zero32
            thr = zero32
            tested = rmode != 0 and n >= min_valid and n > 0
            floor_used = False
            if n > 0 and (tested or wmode == 2):
                for i in range(n):
                    srt[i] = vals[i]
                _isort(srt, n)
                med = _med_sorted(srt, n)
            if tested and rmode == 1:
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
            # noise model v_i = gm / N_i + s_const at the median
            use_v = n > 0 and (wmode == 2 or (tested and rmode == 2))
            gm = 0.0
            s_const = 0.0
            if use_v:
                md = np.float64(med)
                rm = rel * md
                gval = np.float64(grd[v]) if g_on else 0.0
                kg = kappa * gval
                s_const = (rm * rm + kg * kg) + flr * flr
                gm = gain * np.float64(med if med > one else one)
            # 'model': the member(s) nearest the median always stay
            dmin = 0.0
            if tested and rmode == 2:
                dmin = np.inf
                for i in range(n):
                    dev = abs(np.float64(vals[i]) - np.float64(med))
                    if dev < dmin:
                        dmin = dev
            acc = zero32
            num = 0.0
            wsum = 0.0
            m = 0
            n_rej = 0
            self_rej = False
            for i in range(n):
                vi = _div(gm, ncnt[i]) + s_const if use_v else 0.0
                rej = False
                if tested:
                    if rmode == 1:
                        rej = np.float32(abs(np.float32(vals[i] - med))) > thr
                    else:
                        dev = abs(np.float64(vals[i]) - np.float64(med))
                        rej = dev > sigma_d * np.sqrt(vi) and dev > dmin
                if rej:
                    surv[i] = False
                    n_rej += 1
                    if i == self_i:
                        self_rej = True
                else:
                    surv[i] = True
                    if wmode == 0:
                        acc = np.float32(acc + vals[i])
                    else:
                        w = ncnt[i] if wmode == 1 else _div(1.0, vi)
                        ncnt[i] = w
                        num = num + w * np.float64(vals[i])
                        wsum = wsum + w
                    m += 1
            if wmode != 0 and not (wsum > 0.0):
                # no positive weight: the voxel has no result
                for i in range(n):
                    surv[i] = False
                m = 0
            if m > 0:
                res = np.float32(acc / np.float32(m)) if wmode == 0 else np.float32(num / wsum)
            else:
                res = nan32
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
                # cnt_g[g]: number of survivors that read the source of group g
                for i in range(n):
                    cnt_g[i] = 0
                    seen[i] = False
                for i in range(n):
                    if surv[i]:
                        cnt_g[rep[i]] += 1
                # wg[g]: sum of the weights of the survivors that read source g
                if wmode != 0:
                    for i in range(n):
                        wg[i] = 0.0
                    for i in range(n):
                        if surv[i]:
                            wg[rep[i]] = wg[rep[i]] + ncnt[i]
                accv = 0.0
                mm = np.float64(m) * np.float64(m)
                mean = np.float64(res)
                for i in range(n):
                    if not surv[i]:
                        continue
                    g = rep[i]
                    first = not seen[g]
                    seen[g] = True
                    if wmode == 0:
                        fct = np.float64(cnt_g[g]) / mm
                        accv = accv + fct * np.float64(mvar[i])
                    elif first:
                        rr = wg[g] / wsum
                        accv = accv + (rr * rr) * np.float64(mvar[i])
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


@njit(nogil=True, cache=_CACHE)
def _sq0(d):
    """float32 square of d. A square that is not finite gives 0."""
    s = np.float32(d * d)
    if np.isfinite(s):
        return s
    return np.float32(0.0)


@njit(parallel=True, nogil=True, cache=_CACHE)
def gradient_blocks(f, out):
    """Gradient magnitude G of the float32 volume `f` (see `averaging.gradient_magnitude`).

    The operations are the same as in the NumPy and the CUDA code, thus G is
    bit-identical.
    """
    nh, nk, nl = f.shape
    zero = np.float32(0.0)
    for i in prange(nh):
        for j in range(nk):
            for k in range(nl):
                x = f[i, j, k]
                if not np.isfinite(x):
                    out[i, j, k] = zero
                    continue
                fa = _sq0(np.float32(f[i + 1, j, k] - x)) if i + 1 < nh else zero
                fb = _sq0(np.float32(x - f[i - 1, j, k])) if i > 0 else zero
                s0 = fa if fa > fb else fb
                fa = _sq0(np.float32(f[i, j + 1, k] - x)) if j + 1 < nk else zero
                fb = _sq0(np.float32(x - f[i, j - 1, k])) if j > 0 else zero
                s1 = fa if fa > fb else fb
                fa = _sq0(np.float32(f[i, j, k + 1] - x)) if k + 1 < nl else zero
                fb = _sq0(np.float32(x - f[i, j, k - 1])) if k > 0 else zero
                s2 = fa if fa > fb else fb
                out[i, j, k] = np.float32(np.sqrt(np.float32(np.float32(s0 + s1) + s2)))
