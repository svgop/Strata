#!/usr/bin/env python3
"""vq_real_diff.py - bytes-level differential on the REAL MiniMax-M3.1-Flash VQ pack.

Opens one layer's experts GGUF at the release offsets, dumps the REAL GVQ1/GVR8 header
fields of expert 0's w1 (the prime suspect: GVR8's index offset field), decodes the
payload by the strict law (tools/minimax_vq_format.py's Matrix / encode_vq / _gvr_header),
and writes a fixture for the CUDA differential (mm3_vq_test's real-blob mode):

  <out-dir>/layer16.e0.w1.gvq1      the whole GVQ1 payload (verbatim bytes)
  <out-dir>/layer16.e0.r1.gvr8      the whole GVR8 payload (verbatim bytes)
  <out-dir>/layer16.e0.w3.gvq1 / r3.gvr8 / w2.gvq1 / r2.gvr8
  <out-dir>/layer16.cb1.f16 / cb2.f16   the codebooks (verbatim bytes)
  <out-dir>/layer16.e0.w1.ref.f32   the strict-law decode of rows [0, ROWS) in F32
  <out-dir>/layer16.meta            one JSON line: header fields + geometry

Every decode here is written from minimax_vq_format.py's law, NOT from the engine's
reader - an independent witness.  Run:  vq_real_diff.py <release-dir> [layer] [rows]
"""
import json
import pathlib
import struct
import sys
import numpy as np

HEADER = 128
GVQ_HEADER = struct.Struct('<4sHBBIIIIIIQQQQ32s32s')


def align(n):
    return (n + 127) // 128 * 128


def geometry(rows, cols, k):
    if not 0 < rows < 2**32 or not 0 < cols < 2**32 or cols % 64 or k not in (4096, 8192):
        raise ValueError('unsupported GVQ1 shape/K %s' % ((rows, cols, k),))
    elements = rows * cols
    main = align(HEADER + 2 * rows)
    main_end = main + elements * 3 // 16
    plane = align(main_end) if k == 8192 else 0
    end = (plane + elements // 64) if plane else main_end
    return dict(index=main, main_end=main_end, plane=plane, end=end, size=align(end),
                low_stride=cols * 3 // 16, high_stride=cols // 64 if plane else 0)


def res_geometry(rows, cols):
    end = HEADER + rows * (cols // 8)
    return dict(index=HEADER, end=end, size=align(end), stride=cols // 8)


def fp8_table():
    # E4M3FN decode table (minimax_vq_format.py's fp8_table law: 256 f32 values, NaN class at 0x7F/0xFF)
    t = np.zeros(256, dtype='<f4')
    for b in range(256):
        e = (b >> 3) & 15
        m = b & 7
        mag = m * (1.0 / 512.0) if e == 0 else np.ldexp((8 + m) / 8.0, e - 7)
        t[b] = -mag if b & 0x80 else mag
    t[0x7F] = t[0xFF] = np.nan
    return t


def read_gguf(path):
    """Minimal GGUF v3 reader: {name: (type, shape, abs_data_offset, bytes)}."""
    with open(path, 'rb') as f:
        raw = f.read(24)
        magic, ver = struct.unpack('<4sI', raw[:8])
        if magic != b'GGUF':
            raise ValueError('not GGUF: %r' % magic)
        n_tensors, n_kv = struct.unpack('<QQ', raw[8:24])

        def rd_str(f):
            (n,) = struct.unpack('<Q', f.read(8))
            return f.read(n).decode('utf-8')

        def rd_val(f, t):
            if t == 8:   # string
                return rd_str(f)
            if t == 9:   # array
                et, n = struct.unpack('<IQ', f.read(12))
                return [rd_val(f, et) for _ in range(n)]
            fmt = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f', 7: 'B',
                   10: 'Q', 11: 'q', 12: 'd'}[t]
            return struct.unpack('<' + fmt, f.read(struct.calcsize('<' + fmt)))[0]

        kv = {}
        for _ in range(n_kv):
            k = rd_str(f)
            (t,) = struct.unpack('<I', f.read(4))
            kv[k] = rd_val(f, t)
        align_bytes = kv.get('general.alignment', 32)
        infos = []
        for _ in range(n_tensors):
            name = rd_str(f)
            (nd,) = struct.unpack('<I', f.read(4))
            shape = struct.unpack('<%dQ' % nd, f.read(8 * nd))
            (t,) = struct.unpack('<I', f.read(4))
            (off,) = struct.unpack('<Q', f.read(8))
            infos.append((name, t, shape, off))
        data_at = f.tell()
        data_at = (data_at + align_bytes - 1) // align_bytes * align_bytes
        f.seek(0, 2)
        fsize = f.tell()
        return kv, infos, data_at, fsize, path


def main():
    release = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                           '/home/svg/LLM/data/models/VQ/MiniMax-M3.1-Flash')
    layer = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    nrows = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    out = pathlib.Path('/tmp/vq_real_diff')
    out.mkdir(parents=True, exist_ok=True)

    path = release / ('layer-%02d-experts.gguf' % layer)
    kv, infos, data_at, fsize, _ = read_gguf(path)
    print('GGUF %s: %d tensors, %d kv, data@%d' % (path.name, len(infos), len(kv), data_at))

    base = 'language_model.model.layers.%d.block_sparse_moe.' % layer
    names = {n for n, _, _, _ in infos}
    parts = {}
    for proj in ('w1', 'w3', 'w2'):
        for e in (0,):
            for suf, tag in (('.weight', 'g'), ('.weight_residual', 'r')):
                want = base + 'experts.%d.%s%s' % (e, proj, suf)
                if want in names:
                    parts['%s%s' % (proj, tag)] = want
    cb1n = base + 'vq_codebook'
    cb2n = base + 'vq_codebook_residual'
    have_res = cb2n in names
    if not have_res:   # a plain layer: drop the residual tensors the names pass found
        parts = {k: v for k, v in parts.items() if not k.endswith('r')}
    for n in list(parts.values()) + [cb1n]:
        if n not in names:
            raise SystemExit('missing tensor: %s' % n)
    info = {n: (t, shape, off) for n, t, shape, off in infos}

    offs = sorted(off for _, _, _, off in infos) + [fsize - data_at]

    def payload(name):
        _, shape, off = info[name]
        end = offs[offs.index(off) + 1]
        with open(path, 'rb') as f:
            f.seek(data_at + off)
            return f.read(end - off)

    table = fp8_table()
    book1 = np.frombuffer(payload(cb1n), dtype=np.uint8)
    book2 = np.frombuffer(payload(cb2n), dtype=np.uint8) if have_res else np.zeros(2048, dtype=np.uint8)
    print('cb1 %d B, cb2 %s' % (book1.size, ('%d B' % book2.size) if have_res else 'absent (plain layer)'))

    meta = {}
    for proj in ('w1', 'w3', 'w2'):
        gp = np.frombuffer(payload(parts[proj + 'g']), dtype=np.uint8)
        rp = np.frombuffer(payload(parts[proj + 'r']), dtype=np.uint8) if have_res else gp[:128]
        magic, version, bits, vec, rows, cols, k, prec, link, flags, gain, idx, high, total, \
            _sha, _cbname = GVQ_HEADER.unpack(gp[:HEADER].tobytes())
        g = geometry(rows, cols, k)
        print('%s GVQ1 %dx%d k%d v%d: gain=%d idx=%d high=%d total=%d (law %d/%d/%d/%d)'
              % (proj, rows, cols, k, bits, gain, idx, high, total,
                 128, g['index'], g['plane'], g['size']))
        assert magic == b'GVQ1' and version == 1 and k == 8192 and gain == 128 and idx == g['index']
        meta[proj] = dict(rows=int(rows), cols=int(cols), k=int(k), gain=int(gain), idx=int(idx),
                          high=int(high), total=int(total), res_bytes=int(rp.size) if have_res else 0)
        if have_res:
            rmagic, rversion, rbits, rvec, rrows, rcols, rk, rprec, rlink, rflags, rgain, ridx, rhigh, rtotal, \
                _rsha, _rcbname = GVQ_HEADER.unpack(rp[:HEADER].tobytes())
            rg = res_geometry(rrows, rcols)
            print('%s GVR8 %dx%d k%d: u64@32=%d u64@40=%d u64@48=%d total=%d (law: idx@%d size %d)'
                  % (proj, rrows, rcols, rk, rgain, ridx, rhigh, rtotal, rg['index'], rg['size']))
            assert rmagic == b'GVR1' and rgain == rg['index'] == 128 and ridx == 0 and rhigh == 0, \
                'REAL GVR8 header diverges from the strict law'
            meta[proj].update(r_gain=int(rgain), r_idx=int(ridx), r_total=int(rtotal))
        else:
            print('%s plain layer: no GVR8 residual (GVQ1 only)' % proj)

        if proj == 'w1':   # the fixture: expert 0's w1/r1 verbatim + the strict decode of the first rows
            for blob, nm in ((gp, 'w1.gvq1'), (rp, 'r1.gvr8')):
                (out / ('layer%d.e0.%s' % (layer, nm))).write_bytes(blob.tobytes())
            gains = np.frombuffer(gp[gain:gain + 2 * rows].tobytes(), dtype='<f2').astype('<f4')
            nv = cols // 8
            low = gp[g['index']:g['main_end']]
            plane = gp[g['plane']:g['plane'] + g['high_stride'] * nrows] if g['plane'] else None
            res_rows = (rp[128:128 + nrows * nv].reshape(nrows, nv) if have_res else None)
            ref = np.zeros((nrows, cols), dtype='<f4')
            for r in range(nrows):
                i1 = np.zeros(nv, dtype=np.int64)
                base_i = r * nv
                for v in range(nv):
                    j = base_i + v
                    m = j >> 1
                    b0, b1, b2 = low[3 * m], low[3 * m + 1], low[3 * m + 2]
                    lo = (b0 | ((b1 & 15) << 8)) if (j & 1) == 0 else ((b1 >> 4) | (b2 << 4))
                    hi = ((plane[j >> 3] >> (j & 7)) & 1) << 12 if plane is not None else 0
                    i1[v] = lo | hi
                vals = table[book1[(i1[:, None] * 8 + np.arange(8)[None, :]).astype(np.int32)]].astype('<f4')
                if res_rows is not None:
                    r_i = res_rows[r].astype(np.int32)
                    vals += table[book2[(r_i[:, None] * 8 + np.arange(8)[None, :]).astype(np.int32)]].astype('<f4')
                ref[r] = (vals * gains[r]).reshape(cols)
            ref.tofile(out / ('layer%d.e0.w1.ref.f32' % layer))
            (out / ('layer%d.cb1.f16' % layer)).write_bytes(book1.tobytes())
            (out / ('layer%d.cb2.f16' % layer)).write_bytes(book2.tobytes())

    meta = dict(layer=layer, **meta)
    (out / ('layer%d.meta' % layer)).write_text(json.dumps(meta))
    print('fixture written to %s (w1 ref rows %d)' % (out, nrows))
    print('verdict: REAL GVR8 headers carry u64@32=128 (index), u64@40=0, u64@48=0')


if __name__ == '__main__':
    main()
