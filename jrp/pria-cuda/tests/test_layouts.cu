/* test_layouts.cu - host-side checks of the dense layout index maps and the
 * segment geometry (bijectivity, coverage, offset consistency). */
#include <set>
#include <utility>
#include <vector>

#include "layouts.cuh"
#include "test_common.hpp"

using namespace pria;

static void check_maps(i64 T) {
    const i64 P = T / 2, W = T + 1;

    /* column map covers the upper triangle exactly once and round-trips */
    std::set<std::pair<i64, i64>> seen;
    for (i64 j = 0; j < P; ++j)
        for (i64 c = 0; c < W; ++c) {
            i64 s, t;
            col_to_st(T, j, c, s, t);
            CHECK(0 <= s && s <= t && t < T);
            CHECK(seen.insert({s, t}).second);
            i64 j2, c2;
            st_to_col(T, s, t, j2, c2);
            CHECK(j2 == j && c2 == c);
        }
    CHECK((i64)seen.size() == T * (T + 1) / 2);

    /* row map likewise */
    seen.clear();
    for (i64 j = 0; j < P; ++j)
        for (i64 c = 0; c < W; ++c) {
            i64 s, t;
            row_to_st(T, j, c, s, t);
            CHECK(0 <= s && s <= t && t < T);
            CHECK(seen.insert({s, t}).second);
            i64 j2, c2;
            st_to_row(T, s, t, j2, c2);
            CHECK(j2 == j && c2 == c);
        }
    CHECK((i64)seen.size() == T * (T + 1) / 2);
}

static void check_converters(i64 N, i64 T) {
    Dims d = Dims::make(N, T);
    std::vector<char> hit(d.ND, 0);
    for (i64 g = 0; g < d.ND; ++g) {
        i64 gr = colg_to_rowg(d, g);
        CHECK(0 <= gr && gr < d.ND);
        CHECK(rowg_to_colg(d, gr) == g);
        i64 gz = colg_to_d2g(d, g);
        CHECK(0 <= gz && gz < d.ND);
        CHECK(d2g_to_colg(d, gz) == g);
        CHECK(!hit[gr]);
        hit[gr] = 1;
    }
}

template <typename Geom>
static void check_geom(const Geom& geom, const std::vector<int>& offs) {
    const i64 n = geom.d.ND, nseg = geom.num_segments();
    CHECK((i64)offs.size() == nseg + 1);
    CHECK(offs[0] == 0 && offs[nseg] == (int)n);
    i64 prev_seg = -1;
    for (i64 g = 0; g < n; ++g) {
        SegPos p = geom.pos(g);
        CHECK(0 <= p.seg && p.seg < nseg);
        CHECK(p.seg == prev_seg || (p.seg == prev_seg + 1 && p.klocal == 1));
        CHECK((i64)offs[p.seg] == p.start);
        CHECK(p.start + p.klocal - 1 == g);
        CHECK(g < (i64)offs[p.seg + 1]);
        prev_seg = p.seg;
    }
    CHECK(prev_seg == nseg - 1);
}

int main() {
    for (i64 T : {2, 4, 6, 10, 16, 40, 76}) check_maps(T);
    for (auto nt : {std::pair<i64, i64>{1, 2}, {1, 6}, {3, 4}, {5, 10},
                    {7, 16}, {2, 40}})
        check_converters(nt.first, nt.second);

    for (auto nt : {std::pair<i64, i64>{1, 2}, {3, 4}, {5, 10}, {2, 16}}) {
        Dims d = Dims::make(nt.first, nt.second);
        std::vector<int> offs(2 * d.N * d.P + 1);
        build_offsets_xy(d, true, offs.data());
        check_geom(GeomX{d}, offs);
        build_offsets_xy(d, false, offs.data());
        check_geom(GeomY{d}, offs);
        std::vector<int> offz(2 * d.P + 1);
        build_offsets_z(d, offz.data());
        check_geom(GeomZ{d}, offz);
    }

    test_done("test_layouts");
    return 0;
}
