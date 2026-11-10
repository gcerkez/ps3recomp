/*
 * ps3recomp -- native override for the Comet Crash SPURS job 0x187F00
 * (the sprite depth-sort job; comet-crash-re notes, RUNTIME_PATCHES #35).
 *
 * The lifted job's work function (guest func_00000160, called from func_00000054)
 * is three phases over 1024 elements, each phase a lifted loop that keeps every
 * register in ctx->gpr[] and reaches its next block through the SPU_DIRECT
 * trampoline:
 *
 *   1. func_000001B0   per element i: position (x,y,z,1) from the 32-byte
 *                      record at T + 32*i, v = x*R0 + y*R1 + z*R2 + w*R3 for the
 *                      four matrix rows at M+0x00..0x30, then entry E[i] gets
 *                      word0 = v.z and halfword at bytes 4..5 = i.
 *   2. func_00000230   a gnome sort of the 1024 entries by word0 (float,
 *                      descending), capped at 10240 compare steps -- a capped
 *                      sort leaves a partially sorted array, and so must this.
 *   3. func_000002BC   for i in 0..1023, copy the 32-byte record
 *                      T + 32*(E[i] halfword at bytes 4..5) to O + 32*i.
 *
 * This is a transcription of those three loops: the arithmetic goes through the
 * same spu_* helpers (exact SPU float semantics, shuffles, insertion masks), so
 * it is operation-for-operation the lifted math; only where the values live
 * changes (locals instead of ctx->gpr[], no per-block trampoline hop).
 *
 * Memory contract. The lifted loops read and write LS in an interleaved order,
 * which only matters when the regions alias. The override therefore handles the
 * call only when M (the matrix rows), E (entries, 16 KB at r6), T (records,
 * 32 KB at r4) and O (output, 32 KB at r5) are 16-byte aligned (M need not be),
 * inside LS and pairwise disjoint -- the game's job layout -- and returns 0
 * otherwise so the caller runs the lifted function unchanged. Phase 1 stores
 * i into every entry, so phase 3's source indices are always < 1024 and stay in
 * T.
 *
 * Register contract: reads gpr[1] (the insertion mask pattern), gpr[3] (M),
 * gpr[4] (T), gpr[5] (O), gpr[6] (E); writes no gpr. The caller's glue publishes
 * the same exit state the lifted chain leaves (pc, trampoline slot). Everything
 * the lifted chain writes into gpr[2..79] is dead after the call: the only
 * consumer is func_00000054's tail, which reassigns gpr[6..9] and reads gpr[80..82]
 * (never touched here).
 */
#include <stdint.h>
#include <string.h>
#include "spu_context.h"
#include "spu_helpers.h"

#define J_N       1024u
#define J_E_BYTES (J_N * 16u)
#define J_T_BYTES (J_N * 32u)

static inline uint32_t j_ld32(const uint8_t *p) { uint32_t w; memcpy(&w, p, 4); return __builtin_bswap32(w); }
static inline void     j_st32(uint8_t *p, uint32_t v) { v = __builtin_bswap32(v); memcpy(p, &v, 4); }
static inline void     j_st16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)(v >> 8); p[1] = (uint8_t)v; }

static inline u128 j_ld128(const uint8_t *p)
{
    u128 v;
    v._u32[0] = j_ld32(p);     v._u32[1] = j_ld32(p + 4);
    v._u32[2] = j_ld32(p + 8); v._u32[3] = j_ld32(p + 12);
    return v;
}

static int j_overlap(uint32_t a, uint32_t an, uint32_t b, uint32_t bn)
{
    return a < b + bn && b < a + an;
}

typedef struct { uint32_t w[4]; } j_ent;

int ps3_job187f00_compute_native(spu_context *ctx)
{
    const uint32_t M = ctx->gpr[3]._u32[0] & SPU_LS_MASK & ~0xFu;
    const uint32_t T = ctx->gpr[4]._u32[0];
    const uint32_t O = ctx->gpr[5]._u32[0];
    const uint32_t E = ctx->gpr[6]._u32[0];

    if ((T | O | E) & 0xFu) return 0;
    if (T > SPU_LS_SIZE - J_T_BYTES || O > SPU_LS_SIZE - J_T_BYTES ||
        E > SPU_LS_SIZE - J_E_BYTES || M > SPU_LS_SIZE - 0x40u)
        return 0;
    if (j_overlap(M, 0x40, E, J_E_BYTES) || j_overlap(M, 0x40, T, J_T_BYTES) ||
        j_overlap(M, 0x40, O, J_T_BYTES) || j_overlap(E, J_E_BYTES, T, J_T_BYTES) ||
        j_overlap(E, J_E_BYTES, O, J_T_BYTES) || j_overlap(T, J_T_BYTES, O, J_T_BYTES))
        return 0;

    uint8_t *const ls = ctx->ls;

    /* ---- phase 1: depth keys (func_000001B0) ---- */
    /* Only lane 2 of v is kept (rotqbyi 8), and the float helpers are per-lane,
     * so the loop computes just that lane: the same fm/fma/fa operations the
     * lifted code applies to it, on the same operands. The four splat masks are
     * the job's own immediates (word 0/1/2/3 of the position), so lane 2 of
     * x/y/z/w-splat * row is row.z * pos.{x,y,z,w}. */
    const uint32_t R0z = j_ld32(ls + M + 0x00 + 8);
    const uint32_t R1z = j_ld32(ls + M + 0x10 + 8);
    const uint32_t R2z = j_ld32(ls + M + 0x20 + 8);
    const uint32_t R3z = j_ld32(ls + M + 0x30 + 8);
    const u128 one  = spu_ilhu(0x3F80);
    const u128 mIns = spu_cwd(ctx->gpr[1], 12);

    j_ent ent[J_N];   /* job workers run concurrently: stack, not static */

    for (uint32_t i = 0; i < J_N; i++) {
        const u128 orig = j_ld128(ls + E + 16u * i);
        const u128 rec  = j_ld128(ls + T + 32u * i);
        const u128 pos  = spu_shufb(one, rec, mIns);          /* (x, y, z, 1.0f) */
        const uint32_t a = spu__fm_any(R0z, pos._u32[0]);
        const uint32_t b = spu__fm_any(R1z, pos._u32[1]);
        const uint32_t c = spu__fma_any(R2z, pos._u32[2], a, 0, 0);
        const uint32_t d = spu__fma_any(R3z, pos._u32[3], b, 0, 0);
        const uint32_t vz = spu__fa_any(c, d);
        ent[i].w[0] = vz;                                      /* rotqbyi 8: word 2 */
        ent[i].w[1] = (i << 16) | (orig._u32[1] & 0xFFFFu);    /* halfword at bytes 4..5 */
        ent[i].w[2] = orig._u32[2];
        ent[i].w[3] = orig._u32[3];
    }

    /* ---- phase 2: capped gnome sort, descending (func_00000230) ----
     * Indices and the "next unsorted" cursor are the low halfwords of the lifted
     * registers (the upper halfwords are carry-over the guest never reads). */
    {
        uint32_t idx = 1, nxt = 2, cnt = 10240;
        while (idx <= 1023u && cnt != 0) {
            const uint32_t prev = (idx - 1u) & 0xFFFFu;
            u128 ka, kb;
            memset(&ka, 0, sizeof ka); memset(&kb, 0, sizeof kb);
            ka._u32[0] = ent[idx].w[0];
            kb._u32[0] = ent[prev].w[0];
            cnt--;
            if (spu_fcgt(ka, kb)._u32[0] != 0) {
                const j_ent t = ent[idx]; ent[idx] = ent[prev]; ent[prev] = t;
                idx = (idx == 1u) ? 1u : idx - 1u;
            } else {
                idx = nxt;
                nxt = (nxt + 1u) & 0xFFFFu;
            }
        }
    }

    /* ---- write the entries back, then phase 3: gather (func_000002BC) ---- */
    for (uint32_t i = 0; i < J_N; i++) {
        uint8_t *q = ls + E + 16u * i;
        j_st32(q, ent[i].w[0]);     j_st32(q + 4, ent[i].w[1]);
        j_st32(q + 8, ent[i].w[2]); j_st32(q + 12, ent[i].w[3]);
    }
    for (uint32_t i = 0; i < J_N; i++) {
        const uint32_t src = (ent[i].w[1] >> 16) & 0xFFFFu;    /* always < 1024 (phase 1) */
        memcpy(ls + O + 32u * i, ls + T + 32u * src, 32);
    }
    spu_ls_mark(ctx, E, J_E_BYTES);
    spu_ls_mark(ctx, O, J_T_BYTES);
    return 1;
}
