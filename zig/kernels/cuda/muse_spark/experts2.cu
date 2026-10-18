// Muse Spark prompt path: the grouped EXL3 expert kernel (experts_grouped.cuh, unchanged helpers) with 32-member
// tiles: each step's trellis words decoded once for two 16-member groups (two mma accumulator sets). Every member's
// mma, warp order and output are grouped_kernel's, so the bits are its bits; half the decoding for experts with more
// than 16 members (prompt windows of hundreds of rows).

#include "../mimo_v2/experts_grouped.cuh"

namespace muse_x2 {

using namespace tf_exl3x;

template <int CB, int K2, int NT, int PF>
__device__ __forceinline__ void warp_tiles2(const uint32_t* __restrict__ T, int NTILES, int kt0, int nkt, int nt0,
                                            const half* const (&x0)[2], const half* const (&x1)[2],
                                            const bool (&ok0)[2], const bool (&ok1)[2], bool two, int lane,
                                            float (&acc)[2][NT][2][4]) {
    constexpr int TW = Fmt<K2>::TW, LW = Fmt<K2>::LW;
    const LaneMap<K2> map(lane);
    const size_t kstride = (size_t)NTILES * TW;
    const uint32_t* tp = T + ((size_t)kt0 * NTILES + nt0) * TW + lane;

    uint32_t pf[PF][NT][LW];
#pragma unroll
    for (int d = 0; d < PF; ++d)
        if (d < nkt)
#pragma unroll
            for (int i = 0; i < NT; ++i) load_words<K2>(pf[d][i], tp + d * kstride + i * TW, lane);

    for (int ib = 0; ib < nkt; ib += PF) {
#pragma unroll
        for (int d = 0; d < PF; ++d) {
            const int it = ib + d;
            if (it < nkt) {
                uint32_t w[NT][LW];
#pragma unroll
                for (int i = 0; i < NT; ++i)
#pragma unroll
                    for (int l = 0; l < LW; ++l) w[i][l] = pf[d][i][l];
                if (it + PF < nkt)
#pragma unroll
                    for (int i = 0; i < NT; ++i)
                        load_words<K2>(pf[d][i], tp + (size_t)(it + PF) * kstride + i * TW, lane);
                const int k = (kt0 + it) * 16;
                uint32_t a[2][4];
#pragma unroll
                for (int q = 0; q < 2; ++q) {
                    a[q][0] = load_pair(x0[q] + k, ok0[q]);
                    a[q][1] = load_pair(x1[q] + k, ok1[q]);
                    a[q][2] = load_pair(x0[q] + k + 8, ok0[q]);
                    a[q][3] = load_pair(x1[q] + k + 8, ok1[q]);
                }
#pragma unroll
                for (int i = 0; i < NT; ++i) {
                    uint32_t b0[2], b1[2];
                    decode_tile<CB, K2>(w[i], map, lane, b0, b1);
                    mma16816(acc[0][i][0], a[0], b0);
                    mma16816(acc[0][i][1], a[0], b1);
                    if (two) {
                        mma16816(acc[1][i][0], a[1], b0);
                        mma16816(acc[1][i][1], a[1], b1);
                    }
                }
            }
        }
    }
}

// grouped_kernel with 32-member tiles: grid (nexp_max, N / (16 NT), mats * SK * MT2), MT2 = (maxm + 31) / 32.
template <int CB, int NT, int W, int PF, int LO, int HI>
__device__ __forceinline__ void grouped2(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0,
    const int64_t* __restrict__ TP1, const int* __restrict__ K2_0, const int* __restrict__ K2_1,
    const int* __restrict__ uids, const int* __restrict__ ucount, const int* __restrict__ members,
    float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    const int u = blockIdx.x;
    if (u >= ucount[0]) return;
    const int MT = (maxm + 31) / 32;
    const int mtile = blockIdx.z % MT;
    const int split = (blockIdx.z / MT) % SK;
    const int mat = blockIdx.z / MT / SK;
    const half* X = mat ? X1 : X0;
    const int e = uids[u];
    const uint32_t* T = reinterpret_cast<const uint32_t*>(mat ? TP1[e] : TP0[e]);
    const int k2 = mat ? K2_1[e] : K2_0[e];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3;
    const int KT = K >> 4, NTILES = N >> 4;

    __shared__ int rows_sh[32];
    if (threadIdx.x < 32) {
        const int m = mtile * 32 + threadIdx.x;
        const int code = m < maxm ? members[u * maxm + m] : -1;
        rows_sh[threadIdx.x] = code >= 0 ? (code >> 5) * slots + (code & 31) : -1;
    }
    __syncthreads();
    if (rows_sh[0] < 0) return;                           // members come first, so this tile is empty
    const bool two = rows_sh[16] >= 0;
    const half* x0[2];
    const half* x1[2];
    bool ok0[2], ok1[2];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        const int r0 = rows_sh[16 * q + g], r1 = rows_sh[16 * q + g + 8];
        x0[q] = X + (size_t)(r0 < 0 ? 0 : r0) * K + 2 * t;
        x1[q] = X + (size_t)(r1 < 0 ? 0 : r1) * K + 2 * t;
        ok0[q] = r0 >= 0;
        ok1[q] = r1 >= 0;
    }

    const int per_split = KT / SK, per_warp = per_split / W;
    const int kt0 = split * per_split + warp * per_warp;
    const int nt0 = blockIdx.y * NT;

    float acc[2][NT][2][4];
#pragma unroll
    for (int q = 0; q < 2; ++q)
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int c = 0; c < 4; ++c) acc[q][i][h][c] = 0.f;

    switch (k2) {
#define MUSE_X2_CASE(K2_)                                                                                       \
    case K2_:                                                                                                   \
        if constexpr (K2_ >= LO && K2_ <= HI)                                                                   \
            warp_tiles2<CB, K2_, NT, PF>(T, NTILES, kt0, per_warp, nt0, x0, x1, ok0, ok1, two, lane, acc);      \
        else                                                                                                    \
            __trap();                                                                                           \
        break;
        MUSE_X2_CASE(2)
        MUSE_X2_CASE(3)
        MUSE_X2_CASE(4)
        MUSE_X2_CASE(5)
        MUSE_X2_CASE(6)
        MUSE_X2_CASE(7)
        MUSE_X2_CASE(8)
        MUSE_X2_CASE(9)
        MUSE_X2_CASE(10)
        MUSE_X2_CASE(11)
        MUSE_X2_CASE(12)
        MUSE_X2_CASE(13)
        MUSE_X2_CASE(14)
        MUSE_X2_CASE(15)
        MUSE_X2_CASE(16)
#undef MUSE_X2_CASE
        default:
            __trap();
    }

    // each group's warps' partial sums through shared memory, added in warp order (grouped_kernel's)
    __shared__ float red[W][16][NT * 16];
#pragma unroll
    for (int q = 0; q < 2; ++q) {
        if (q == 1 && !two) break;
        if (q) __syncthreads();
#pragma unroll
        for (int i = 0; i < NT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                const int col = i * 16 + h * 8 + 2 * t;
                red[warp][g][col] = acc[q][i][h][0];
                red[warp][g][col + 1] = acc[q][i][h][1];
                red[warp][g + 8][col] = acc[q][i][h][2];
                red[warp][g + 8][col + 1] = acc[q][i][h][3];
            }
        __syncthreads();
        for (int idx = threadIdx.x; idx < 16 * NT * 16; idx += W * 32) {
            const int row = idx / (NT * 16), col = idx % (NT * 16);
            const int r = rows_sh[16 * q + row];
            if (r < 0) continue;
            float s = red[0][row][col];
#pragma unroll
            for (int w = 1; w < W; ++w) s += red[w][row][col];
            Z[(((size_t)mat * SK + split) * P + r) * N + nt0 * 16 + col] = s;
        }
    }
}

}  // namespace muse_x2

extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_1_8_8(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 1, 8, 8>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_1_2_10(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 1, 2, 10>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_1_2_16(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 1, 2, 16>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_2_8_8(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 2, 8, 8>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_2_2_10(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 2, 2, 10>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_8_4_2_2_16(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 8, 4, 2, 2, 16>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_4_4_2_8_8(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 4, 4, 2, 8, 8>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_4_4_2_2_10(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 4, 4, 2, 2, 10>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
extern "C" __global__ void __launch_bounds__(128) muse_grouped2_4_4_2_2_16(
    const half* __restrict__ X0, const half* __restrict__ X1, const int64_t* __restrict__ TP0, const int64_t* __restrict__ TP1,
    const int* __restrict__ K2_0, const int* __restrict__ K2_1, const int* __restrict__ uids, const int* __restrict__ ucount,
    const int* __restrict__ members, float* __restrict__ Z, int K, int N, int P, int SK, int maxm, int slots) {
    muse_x2::grouped2<2, 4, 4, 2, 2, 16>(X0, X1, TP0, TP1, K2_0, K2_1, uids, ucount, members, Z, K, N, P, SK, maxm, slots);
}
