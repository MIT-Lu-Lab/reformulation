/* layouts.cuh - dense pair-packed layouts for upper-triangular JRP variables.
 *
 * All variables live on the upper triangle {(s, t) : 0 <= s <= t < T} of a
 * T x T matrix per item, with T even (the solver pads odd T). The triangle is
 * stored densely by pairing complementary columns/rows so every dense row is
 * exactly full, mirroring the JAX implementation's densify_columnwise,
 * densify_rowwise and densify_tensor_dim2:
 *
 *   P = T/2 pairs, W = T+1 dense width, D = P*W = T(T+1)/2 entries per item.
 *
 * Column layout (primal x, holding costs h), buffer index (i*P + j)*W + c:
 *   c <= j : column t = j, entry s = c            (j+1 entries)
 *   c >  j : column t = T-1-j, entry s = T-c      (T-j entries, s descending)
 *
 * Row layout (item dual y), buffer index (i*P + j)*W + c:
 *   c <= T-1-j : row s = j, entry t = T-1-c       (T-j entries, t descending)
 *   c >= T-j   : row s = T-1-j, entry t = c-1     (j+1 entries)
 *
 * Dim2 layout (joint dual z), buffer index (j*W + c)*N + i: the row layout's
 * (j, c) -> (s, t) map with the item index i as the fastest dimension, so the
 * per-period groups {(i, t) : t >= s, i in [N]} are contiguous.
 *
 * Each dense row (column/row layouts) or dense slice (dim2 layout) is the
 * concatenation of exactly two projection segments, so the whole buffer is a
 * contiguous sequence of segments - the property the CUB segmented sort in
 * the projections relies on.
 */
#ifndef PRIA_LAYOUTS_CUH
#define PRIA_LAYOUTS_CUH

#include <cstdint>

#ifdef __CUDACC__
#define PRIA_HD __host__ __device__ __forceinline__
#else
#define PRIA_HD inline
#endif

namespace pria {

using i64 = int64_t;

/* Static geometry of one padded instance. */
struct Dims {
    i64 N;  /* items */
    i64 T;  /* padded (even) time periods */
    i64 P;  /* T/2 dense rows per item */
    i64 W;  /* T+1 dense width */
    i64 D;  /* P*W = T(T+1)/2 entries per item */
    i64 ND; /* N*D entries per variable class */

    static Dims make(i64 n, i64 t_padded) {
        Dims d;
        d.N = n;
        d.T = t_padded;
        d.P = t_padded / 2;
        d.W = t_padded + 1;
        d.D = d.P * d.W;
        d.ND = n * d.D;
        return d;
    }
};

/* ---- (j, c) <-> (s, t) maps -------------------------------------------- */

PRIA_HD void col_to_st(i64 T, i64 j, i64 c, i64& s, i64& t) {
    if (c <= j) { s = c;     t = j; }
    else        { s = T - c; t = T - 1 - j; }
}

PRIA_HD void st_to_col(i64 T, i64 s, i64 t, i64& j, i64& c) {
    if (t < (T >> 1)) { j = t;         c = s; }
    else              { j = T - 1 - t; c = T - s; }
}

PRIA_HD void row_to_st(i64 T, i64 j, i64 c, i64& s, i64& t) {
    if (c <= T - 1 - j) { s = j;         t = T - 1 - c; }
    else                { s = T - 1 - j; t = c - 1; }
}

PRIA_HD void st_to_row(i64 T, i64 s, i64 t, i64& j, i64& c) {
    if (s < (T >> 1)) { j = s;         c = T - 1 - t; }
    else              { j = T - 1 - s; c = t + 1; }
}

/* ---- buffer-index converters ------------------------------------------- */

/* col-layout index -> row-layout index of the same (i, s, t) */
PRIA_HD i64 colg_to_rowg(const Dims d, i64 g) {
    i64 i = g / d.D, r = g - i * d.D, j = r / d.W, c = r - j * d.W;
    i64 s, t, j2, c2;
    col_to_st(d.T, j, c, s, t);
    st_to_row(d.T, s, t, j2, c2);
    return (i * d.P + j2) * d.W + c2;
}

/* col-layout index -> dim2-layout index of the same (i, s, t) */
PRIA_HD i64 colg_to_d2g(const Dims d, i64 g) {
    i64 i = g / d.D, r = g - i * d.D, j = r / d.W, c = r - j * d.W;
    i64 s, t, j2, c2;
    col_to_st(d.T, j, c, s, t);
    st_to_row(d.T, s, t, j2, c2);
    return (j2 * d.W + c2) * d.N + i;
}

/* row-layout index -> col-layout index of the same (i, s, t) */
PRIA_HD i64 rowg_to_colg(const Dims d, i64 g) {
    i64 i = g / d.D, r = g - i * d.D, j = r / d.W, c = r - j * d.W;
    i64 s, t, j2, c2;
    row_to_st(d.T, j, c, s, t);
    st_to_col(d.T, s, t, j2, c2);
    return (i * d.P + j2) * d.W + c2;
}

/* dim2-layout index -> col-layout index of the same (i, s, t) */
PRIA_HD i64 d2g_to_colg(const Dims d, i64 g) {
    i64 i = g % d.N, jc = g / d.N, j = jc / d.W, c = jc - j * d.W;
    i64 s, t, j2, c2;
    row_to_st(d.T, j, c, s, t);
    st_to_col(d.T, s, t, j2, c2);
    return (i * d.P + j2) * d.W + c2;
}

/* ---- projection segment geometry --------------------------------------- *
 *
 * Segments are numbered in buffer order. Each geometry knows, for a buffer
 * index g: its segment id, its 1-based position inside the segment, the
 * buffer index where the segment starts, and the segment's simplex target.
 */

struct SegPos {
    i64 seg;      /* segment id */
    i64 klocal;   /* 1-based rank of g inside its segment */
    i64 start;    /* buffer index of the segment start */
};

/* x: per (i, t) column simplex, target 1. Segments per dense row: [0, j+1)
 * (column t = j) and [j+1, W) (column t = T-1-j). */
struct GeomX {
    static constexpr bool kRowwise = true; /* two segments per dense row */
    static constexpr bool kIsX = true;
    Dims d;
    PRIA_HD SegPos pos(i64 g) const {
        i64 grp = g / d.W, c = g - grp * d.W, j = grp % d.P;
        i64 split = j + 1;
        bool hi = c >= split;
        SegPos p;
        p.seg = 2 * grp + (hi ? 1 : 0);
        p.klocal = hi ? (c - split + 1) : (c + 1);
        p.start = g - p.klocal + 1;
        return p;
    }
    template <typename real>
    PRIA_HD real target(i64 /*seg*/, const real* /*k*/, real /*k0*/) const {
        return real(1);
    }
    i64 num_segments() const { return 2 * d.N * d.P; }
};

/* y: per (i, s) row simplex, target K_i. Segments per dense row: [0, T-j)
 * (row s = j) and [T-j, W) (row s = T-1-j). */
struct GeomY {
    static constexpr bool kRowwise = true;
    static constexpr bool kIsX = false;
    Dims d;
    PRIA_HD SegPos pos(i64 g) const {
        i64 grp = g / d.W, c = g - grp * d.W, j = grp % d.P;
        i64 split = d.T - j;
        bool hi = c >= split;
        SegPos p;
        p.seg = 2 * grp + (hi ? 1 : 0);
        p.klocal = hi ? (c - split + 1) : (c + 1);
        p.start = g - p.klocal + 1;
        return p;
    }
    template <typename real>
    PRIA_HD real target(i64 seg, const real* k, real /*k0*/) const {
        return k[(seg >> 1) / d.P];
    }
    i64 num_segments() const { return 2 * d.N * d.P; }
};

/* z: per s joint simplex over {(i, t) : t >= s}, target K0. Segments per
 * dense slice j: [0, (T-j)*N) (period s = j) and the rest (s = T-1-j). */
struct GeomZ {
    static constexpr bool kRowwise = false; /* segments span all items */
    static constexpr bool kIsX = false;
    Dims d;
    PRIA_HD SegPos pos(i64 g) const {
        i64 slice = d.W * d.N;
        i64 j = g / slice, l = g - j * slice;
        i64 split = (d.T - j) * d.N;
        bool hi = l >= split;
        SegPos p;
        p.seg = 2 * j + (hi ? 1 : 0);
        p.klocal = hi ? (l - split + 1) : (l + 1);
        p.start = g - p.klocal + 1;
        return p;
    }
    template <typename real>
    PRIA_HD real target(i64 /*seg*/, const real* /*k*/, real k0) const {
        return k0;
    }
    i64 num_segments() const { return 2 * d.P; }
};

/* Host-side segment offset builders (offsets[seg] = start, offsets[nseg] =
 * total items). Segments tile the buffer, so end[i] == begin[i+1]. */
inline void build_offsets_xy(const Dims& d, bool is_x, int* offs) {
    i64 seg = 0;
    for (i64 i = 0; i < d.N; ++i)
        for (i64 j = 0; j < d.P; ++j) {
            i64 base = (i * d.P + j) * d.W;
            i64 split = is_x ? (j + 1) : (d.T - j);
            offs[seg++] = (int)base;
            offs[seg++] = (int)(base + split);
        }
    offs[seg] = (int)d.ND;
}

inline void build_offsets_z(const Dims& d, int* offs) {
    i64 seg = 0;
    for (i64 j = 0; j < d.P; ++j) {
        i64 base = j * d.W * d.N;
        offs[seg++] = (int)base;
        offs[seg++] = (int)(base + (d.T - j) * d.N);
    }
    offs[seg] = (int)d.ND;
}

} // namespace pria

#endif /* PRIA_LAYOUTS_CUH */
