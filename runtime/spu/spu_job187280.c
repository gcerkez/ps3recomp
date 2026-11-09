/* spu_job187280.c -- native host override for Comet Crash's SPURS geometry
 * batch job (guest binary at EA 0x187280, workload image 4).
 *
 * Title-specific host code in the fork runtime, following the
 * spu_mp3_codec.c precedent. The generated lift for the job calls the work
 * function spu_job_00187280_spu_func_000004A0 from inside
 * spu_job_00187280_spu_func_0000029C; the title build patches that one call
 * site to cc_job187280_compute() (workspace glue), which forwards here
 * unless PS3_NO_JOB_OVERRIDE is set, in which case it calls the lifted
 * function exactly as before.
 *
 * What the work function computes (derived field-by-field from the lift;
 * see notes 050): per group, compose two 4-row transform sets from a
 * 64-byte group record and a shared 64-byte block; then either a transform
 * pass (vertex transform + SPU-rsqrt normal normalize, writing a 16-byte
 * position stream and a 32-byte record stream) or a dot-product pass
 * (16-byte stream), selected by the mode argument; then a triangle
 * frustum-cull pass over an index stream, writing 12-byte records.
 *
 * Exactness contract: this is a mechanical transcription of the lifted
 * compute (guest funcs 0x4A0..0xB00), operation for operation, using the
 * fork's own spu_* helpers -- including the LUT-based frsqest/frest
 * estimates and the spu_fi interpolation step -- with no mathematically
 * "equivalent" simplifications. The override writes only guest LS and
 * leaves ctx->gpr untouched: every register the job reads after the work
 * call (the PUT-stream cursors/sizes in gpr[80..89], the stack-saved
 * registers restored by the epilogue) is set before the call and is never
 * written by the compute, so the lifted job's PUT phases, epilogue and
 * all runtime staging/mailbox/chain bookkeeping run exactly as before.
 *
 * Parallelization (all cursor arithmetic below is affine in the element /
 * triangle index, so chunk results are identical to the sequential run):
 *  - Groups run sequentially: the 16-byte position stream base (gpr[18])
 *    is per-group scratch -- every group re-reads the input stream and
 *    rewrites that region, and its cull pass consumes it.
 *  - Within a group, transform/dot elements fan out across a host thread
 *    pool: per-element outputs are disjoint (16 B / 32 B strides) and the
 *    only loop-carried registers are the three cursors and the counter,
 *    all affine in the element index.
 *  - The cull pass splits into a parallel test phase (0x820 + the 0xA70
 *    refine tests; pure reads + register math) and a serial write phase.
 *    The split is necessary: gpr[13] is loop-carried through the 0xB00
 *    write path, and the 12-byte record stride makes neighbouring
 *    records share 16-byte LS words, so the read-modify-write merges in
 *    the write paths must stay in order. The test phase is where the
 *    floating-point cost lives.
 */
#include "spu_context.h"
#include "spu_helpers.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* ------------------------------------------------------------------ */
/* Small reentrant thread pool: callers submit one range job at a     */
/* time and help drain it; workers always help on the queue head.     */
/* Multiple concurrent callers (several SPU threads running this job  */
/* at once) each get their own job object on their own stack.         */
/* ------------------------------------------------------------------ */

typedef void (*jchunk_fn)(void *arg, uint32_t begin, uint32_t end);

typedef struct jjob {
    jchunk_fn      fn;
    void          *arg;
    uint32_t       n_items;
    uint32_t       chunk;      /* items per chunk */
    uint32_t       n_chunks;
    uint32_t       next;       /* next chunk index to hand out */
    uint32_t       done;       /* chunks finished */
    uint32_t       active;     /* threads currently draining this job */
    struct jjob   *qnext;
    pthread_cond_t finished;   /* broadcast when done==n_chunks && active==0 */
} jjob_t;

static pthread_mutex_t j_pool_mtx  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  j_pool_work = PTHREAD_COND_INITIALIZER;
static jjob_t         *j_pool_head = 0, *j_pool_tail = 0;
static int             j_pool_nworkers = 0;
static int             j_pool_ready = 0;

static void j_drain(jjob_t *j)
{
    for (;;) {
        pthread_mutex_lock(&j_pool_mtx);
        if (j->next >= j->n_chunks) {
            pthread_mutex_unlock(&j_pool_mtx);
            return;
        }
        uint32_t ci = j->next++;
        pthread_mutex_unlock(&j_pool_mtx);

        uint32_t b = ci * j->chunk;
        uint32_t e = b + j->chunk;
        if (e > j->n_items) e = j->n_items;
        j->fn(j->arg, b, e);

        pthread_mutex_lock(&j_pool_mtx);
        j->done++;
        pthread_mutex_unlock(&j_pool_mtx);
    }
}

static void *j_worker(void *unused)
{
    (void)unused;
    pthread_mutex_lock(&j_pool_mtx);
    for (;;) {
        jjob_t *j = j_pool_head;
        if (!j || j->done == j->n_chunks) {
            /* Nothing to help with: no job queued, or the head job is
             * complete and its owner is about to unlink it (the owner
             * broadcasts j_pool_work when it does). A worker must never
             * re-take a completed job: its chunks are gone, so draining
             * it is a no-op that would spin against the owner's wait
             * and starve the pool mutex (found by the harness, which
             * livelocked at full CPU). */
            pthread_cond_wait(&j_pool_work, &j_pool_mtx);
            continue;
        }
        j->active++;
        pthread_mutex_unlock(&j_pool_mtx);
        j_drain(j);
        pthread_mutex_lock(&j_pool_mtx);
        j->active--;
        if (j->done == j->n_chunks && j->active == 0)
            pthread_cond_broadcast(&j->finished);
    }
    return 0;
}

static void j_pool_init(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    int want = (n > 1) ? (int)n - 1 : 0;
    if (want > 15) want = 15;
    int made = 0;
    for (int i = 0; i < want; i++) {
        pthread_t th;
        if (pthread_create(&th, 0, j_worker, 0) != 0)
            break;
        pthread_detach(th);
        made++;
    }
    j_pool_nworkers = made;
    j_pool_ready = made > 0;
}

/* Run fn(arg, begin, end) over [0, n) in parallel chunks. The calling
 * thread participates. Falls back to a direct call when the pool is
 * unavailable or the range is too small to be worth dispatching. */
static void j_pool_run(jchunk_fn fn, void *arg, uint32_t n)
{
    if (n == 0)
        return;
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, j_pool_init);
    if (!j_pool_ready || n < 256) {
        fn(arg, 0, n);
        return;
    }
    uint32_t target = (uint32_t)(j_pool_nworkers + 1) * 2;
    uint32_t chunk = (n + target - 1) / target;
    if (chunk == 0) chunk = 1;

    jjob_t j;
    memset(&j, 0, sizeof j);
    j.fn = fn; j.arg = arg; j.n_items = n; j.chunk = chunk;
    j.n_chunks = (n + chunk - 1) / chunk;
    pthread_cond_init(&j.finished, 0);

    pthread_mutex_lock(&j_pool_mtx);
    if (j_pool_tail) j_pool_tail->qnext = &j; else j_pool_head = &j;
    j_pool_tail = &j;
    pthread_cond_broadcast(&j_pool_work);
    pthread_mutex_unlock(&j_pool_mtx);

    j_drain(&j);

    pthread_mutex_lock(&j_pool_mtx);
    while (!(j.done == j.n_chunks && j.active == 0))
        pthread_cond_wait(&j.finished, &j_pool_mtx);
    /* unlink (only safe now: no worker can touch the job any more) */
    jjob_t **pp = &j_pool_head;
    while (*pp && *pp != &j) pp = &(*pp)->qnext;
    if (*pp) *pp = j.qnext;
    if (j_pool_tail == &j) {
        j_pool_tail = 0;
        for (jjob_t *q = j_pool_head; q; q = q->qnext) j_pool_tail = q;
    }
    /* wake workers parked on the completed head so they move on to
     * whatever is queued behind it */
    pthread_cond_broadcast(&j_pool_work);
    pthread_mutex_unlock(&j_pool_mtx);
    pthread_cond_destroy(&j.finished);
}

/* ------------------------------------------------------------------ */
/* Transcription. G[] is a private copy of the register file; ctx is  */
/* used only for LS access (ctx->ls), which is how the lifted code    */
/* addresses it too. Register numbers match the lifted code exactly.  */
/* ------------------------------------------------------------------ */

/* Guest func_000004A0 prologue: argument saves, LS constants, the
 * gpr[43] product, gates. Returns 0 when the guest returns immediately
 * (group count <= 0) -- in that case it has touched no LS. */
static int j_prologue(spu_context *ctx, u128 *G)
{
    G[54] = spu_ori(G[4], 0);
    G[27] = spu_shlqbyi(G[5], 0);
    G[4]  = spu_ori(G[3], 0);
    G[55] = spu_shlqbyi(G[6], 0);
    G[2]  = spu_cgti(G[54], 0);
    G[41] = spu_shlqbyi(G[7], 0);
    G[60] = spu_ori(G[8], 0);
    G[34] = spu_shlqbyi(G[9], 0);
    G[62] = spu_ori(G[11], 0);
    G[61] = spu_shlqbyi(G[12], 0);
    G[42] = spu_ori(G[15], 0);
    if (G[2]._u32[0] == 0) return 0;                    /* SPU_RET */
    G[6]  = spu_mpyh(G[3], G[5]);
    G[59] = spu_ls_read128(ctx, 0xB50);
    G[3]  = spu_mpyh(G[5], G[3]);
    G[50] = spu_ls_read128(ctx, 0xB60);
    G[4]  = spu_mpyu(G[4], G[5]);
    G[51] = spu_ls_read128(ctx, 0xB70);
    G[47] = spu_ilhu(0x405);
    G[23] = spu_shlqbyi(G[10], 0);
    G[48] = spu_ilhu(0x809);
    G[44] = spu_shlqbyi(G[14], 0);
    G[49] = spu_ilhu(0xC0D);
    G[53] = spu_fsmbi(0x0);
    G[58] = spu_and(G[55], G[13]);
    G[45] = spu_fsmbi(0x0);
    G[47] = spu_iohl(G[47], 0x607);
    G[52] = spu_fsmbi(0x0);
    G[5]  = spu_a(G[6], G[3]);
    G[48] = spu_iohl(G[48], 0xA0B);
    G[43] = spu_a(G[5], G[4]);
    G[49] = spu_iohl(G[49], 0xE0F);
    G[56] = spu_cgti(G[27], 0);
    G[57] = spu_cgti(G[7], 0);
    G[46] = spu_ila(0x10203);
    return 1;
}

/* Guest func_00000538, first part: compose the two transform row sets
 * (results in G[19..22] and G[29..32]) from the group record at G[23]
 * and the shared block at G[42]. The branch at its end (mode flag /
 * element-count gate) is the caller's. */
static void j_compose(spu_context *ctx, u128 *G)
{
    G[22] = spu_ls_read128(ctx, G[23]._u32[0] + 0x10);
    G[75] = spu_ls_read128(ctx, G[23]._u32[0] + 0x0);
    G[76] = spu_ls_read128(ctx, G[23]._u32[0] + 0x20);
    G[69] = spu_ls_read128(ctx, G[42]._u32[0] + 0x0);
    G[64] = spu_ls_read128(ctx, G[42]._u32[0] + 0x10);
    G[30] = spu_ls_read128(ctx, G[42]._u32[0] + 0x20);
    G[74] = spu_shufb(G[22], G[59], G[50]);
    G[21] = spu_shufb(G[22], G[59], G[51]);
    G[20] = spu_shufb(G[75], G[76], G[51]);
    G[19] = spu_shufb(G[75], G[76], G[50]);
    G[28] = spu_ls_read128(ctx, G[42]._u32[0] + 0x30);
    G[22] = spu_shufb(G[20], G[21], G[51]);
    G[21] = spu_shufb(G[20], G[21], G[50]);
    G[20] = spu_shufb(G[19], G[74], G[51]);
    G[19] = spu_shufb(G[19], G[74], G[50]);
    G[73] = spu_shufb(G[22], G[22], G[46]);
    G[72] = spu_shufb(G[21], G[21], G[46]);
    G[71] = spu_shufb(G[20], G[20], G[46]);
    G[70] = spu_shufb(G[19], G[19], G[46]);
    G[16] = spu_fm(G[69], G[73]);
    G[68] = spu_shufb(G[19], G[19], G[47]);
    G[15] = spu_fm(G[69], G[72]);
    G[67] = spu_shufb(G[20], G[20], G[47]);
    G[37] = spu_fm(G[69], G[71]);
    G[66] = spu_shufb(G[21], G[21], G[47]);
    G[63] = spu_fm(G[69], G[70]);
    G[65] = spu_shufb(G[22], G[22], G[47]);
    G[39] = spu_fm(G[64], G[68]);
    G[32] = spu_shufb(G[21], G[21], G[48]);
    G[35] = spu_fm(G[64], G[67]);
    G[31] = spu_shufb(G[22], G[22], G[48]);
    G[10] = spu_fm(G[64], G[66]);
    G[29] = spu_shufb(G[22], G[22], G[49]);
    G[12] = spu_fm(G[64], G[65]);
    G[40] = spu_shufb(G[19], G[19], G[48]);
    G[9]  = spu_fma(G[30], G[32], G[15]);
    G[38] = spu_shufb(G[19], G[19], G[49]);
    G[13] = spu_fma(G[30], G[31], G[16]);
    G[36] = spu_shufb(G[20], G[20], G[48]);
    G[33] = spu_shufb(G[20], G[20], G[49]);
    G[8]  = spu_fma(G[30], G[40], G[63]);
    G[17] = spu_shufb(G[21], G[21], G[49]);
    G[26] = spu_fma(G[28], G[38], G[39]);
    G[25] = spu_fma(G[30], G[36], G[37]);
    G[11] = spu_fma(G[28], G[29], G[12]);
    G[24] = spu_fma(G[28], G[33], G[35]);
    G[14] = spu_fma(G[28], G[17], G[10]);
    G[30] = spu_fa(G[8], G[26]);
    G[32] = spu_fa(G[13], G[11]);
    G[31] = spu_fa(G[25], G[24]);
    G[29] = spu_fa(G[9], G[14]);
}

/* Guest func_00000538 tail (transform mode, elements > 0): loop setup.
 * After this, G[14] = input cursor, G[16] = 16-byte output cursor,
 * G[15] = 32-byte output cursor, G[17] = element counter (0). */
static void j_xform_setup(u128 *G)
{
    G[78] = spu_shli(G[53], 5);
    G[26] = spu_ilhu(0x405);
    G[38] = spu_cwd(G[1], 0);
    G[25] = spu_ilhu(0x809);
    G[36] = spu_cwd(G[1], 4);
    G[24] = spu_ilhu(0xC0D);
    G[37] = spu_cwd(G[1], 8);
    G[77] = spu_ilhu(0x3F80);
    G[35] = spu_cwd(G[1], 12);
    G[15] = spu_a(G[78], G[62]);
    G[14] = spu_shlqbyi(G[60], 0);
    G[26] = spu_iohl(G[26], 0x607);
    G[17] = spu_fsmbi(0x0);
    G[25] = spu_iohl(G[25], 0xA0B);
    G[16] = spu_shlqbyi(G[18], 0);
    G[24] = spu_iohl(G[24], 0xE0F);
    G[28] = spu_ori(G[77], 1);
    G[40] = spu_ila(0x10203);
    G[33] = spu_ila(0x10203);
    G[39] = spu_ilhu(0x3F00);
}

/* Guest func_0000095C (dot-product mode, elements > 0): loop setup.
 * After this, G[12] = input cursor, G[13] = 16-byte output cursor,
 * G[14] = element counter (0). */
static void j_dot_setup(u128 *G)
{
    G[17] = spu_ilhu(0x405);
    G[16] = spu_ilhu(0x809);
    G[21] = spu_cwd(G[1], 0);
    G[15] = spu_ilhu(0xC0D);
    G[20] = spu_cwd(G[1], 4);
    G[17] = spu_iohl(G[17], 0x607);
    G[19] = spu_cwd(G[1], 8);
    G[16] = spu_iohl(G[16], 0xA0B);
    G[12] = spu_shlqbyi(G[60], 0);
    G[15] = spu_iohl(G[15], 0xE0F);
    G[14] = spu_fsmbi(0x0);
    G[24] = spu_ila(0x10203);
    G[13] = spu_shlqbyi(G[18], 0);
    G[22] = spu_ila(0x10203);
}

/* Guest func_00000670: one transform-pass element. Cursors on entry:
 * G[14] input (+32/element), G[16] 16-byte out (+16), G[15] 32-byte
 * out (+32), G[17] counter (pre-increment value). The guest loop's
 * ceq exit test is subsumed by the caller's element range. */
static void j_xform_elem(spu_context *ctx, u128 *G)
{
    G[74] = spu_il(12);
    G[9]  = spu_ls_read128(ctx, G[14]._u32[0] + 0x10);
    G[71] = spu_ai(G[14], 12);
    G[67] = spu_fsmbi(0x0);
    G[72] = spu_il(20);
    G[70] = spu_ls_read128(ctx, G[14]._u32[0] + G[74]._u32[0]);
    G[11] = spu_ai(G[14], 20);
    G[73] = spu_ls_read128(ctx, G[14]._u32[0] + 0x0);
    G[68] = spu_il(4);
    G[3]  = spu_ls_read128(ctx, G[14]._u32[0] + G[72]._u32[0]);
    G[65] = spu_ilhu(0x3F80);
    G[64] = spu_il(8);
    G[10] = spu_rotqby(G[9], G[14]);
    G[63] = spu_il(60);
    G[6]  = spu_ls_read128(ctx, G[14]._u32[0] + G[68]._u32[0]);
    G[2]  = spu_ai(G[14], 4);
    G[66] = spu_rotqby(G[70], G[71]);
    G[78] = spu_ai(G[14], 8);
    G[8]  = spu_rotqby(G[73], G[14]);
    G[77] = spu_ai(G[23], 60);
    G[4]  = spu_rotqby(G[3], G[11]);
    G[17] = spu_ai(G[17], 1);
    G[12] = spu_ls_read128(ctx, G[14]._u32[0] + G[64]._u32[0]);
    G[5]  = spu_shufb(G[66], G[67], G[38]);
    G[13] = spu_shufb(G[8], G[65], G[38]);
    G[76] = spu_ls_read128(ctx, G[23]._u32[0] + G[63]._u32[0]);
    G[69] = spu_rotqby(G[6], G[2]);
    G[7]  = spu_shufb(G[10], G[5], G[36]);
    G[79] = spu_rotqby(G[12], G[78]);
    G[12] = spu_ai(G[14], 28);
    G[75] = spu_shufb(G[69], G[13], G[36]);
    G[69] = spu_ai(G[14], 24);
    G[68] = spu_shufb(G[4], G[7], G[37]);
    G[73] = spu_rotqby(G[76], G[77]);
    G[70] = spu_shufb(G[79], G[75], G[37]);
    G[79] = spu_ceq(G[27], G[17]);
    G[74] = spu_shufb(G[68], G[68], G[33]);
    G[72] = spu_shufb(G[68], G[68], G[26]);
    G[64] = spu_shufb(G[68], G[68], G[24]);
    G[66] = spu_shufb(G[68], G[68], G[25]);
    G[67] = spu_fm(G[19], G[74]);
    G[65] = spu_fm(G[20], G[72]);
    G[71] = spu_shufb(G[73], G[73], G[40]);
    G[72] = spu_il(24);
    G[63] = spu_fma(G[68], G[71], G[70]);
    G[9]  = spu_fma(G[21], G[66], G[67]);
    G[8]  = spu_fma(G[22], G[64], G[65]);
    G[10] = spu_shufb(G[63], G[63], G[33]);
    G[13] = spu_shufb(G[63], G[63], G[26]);
    G[9]  = spu_fa(G[9], G[8]);
    G[11] = spu_shufb(G[63], G[63], G[24]);
    G[3]  = spu_shufb(G[63], G[63], G[25]);
    G[2]  = spu_fm(G[30], G[10]);
    G[5]  = spu_fm(G[31], G[13]);
    G[6]  = spu_fm(G[19], G[10]);
    G[7]  = spu_fm(G[20], G[13]);
    G[4]  = spu_fm(G[9], G[9]);
    G[63] = spu_il(28);
    G[78] = spu_rotqbyi(G[9], 4);
    G[76] = spu_fma(G[29], G[3], G[2]);
    G[77] = spu_fma(G[32], G[11], G[5]);
    G[74] = spu_fma(G[21], G[3], G[6]);
    G[75] = spu_fma(G[22], G[11], G[7]);
    G[70] = spu_fma(G[78], G[78], G[4]);
    G[73] = spu_fa(G[76], G[77]);
    G[67] = spu_fa(G[74], G[75]);
    G[71] = spu_rotqbyi(G[70], 8);
    spu_ls_write128(ctx, G[16]._u32[0] + 0x0, G[73]);
    G[10] = spu_fa(G[70], G[71]);
    G[16] = spu_ai(G[16], 16);
    G[68] = spu_ls_read128(ctx, G[14]._u32[0] + G[72]._u32[0]);
    G[65] = spu_frsqest(G[10]);
    G[66] = spu_rotqby(G[68], G[69]);
    G[11] = spu_fi(G[10], G[65]);
    G[64] = spu_shufb(G[66], G[67], G[35]);
    spu_ls_write128(ctx, G[15]._u32[0] + 0x0, G[64]);
    G[8]  = spu_ls_read128(ctx, G[14]._u32[0] + G[63]._u32[0]);
    G[13] = spu_fm(G[11], G[10]);
    G[3]  = spu_fm(G[11], G[39]);
    G[14] = spu_ai(G[14], 32);
    G[6]  = spu_rotqby(G[8], G[12]);
    G[7]  = spu_fnms(G[11], G[13], G[28]);
    G[4]  = spu_fma(G[7], G[3], G[11]);
    G[5]  = spu_fm(G[9], G[4]);
    G[2]  = spu_shufb(G[6], G[5], G[35]);
    spu_ls_write128(ctx, G[15]._u32[0] + 0x10, G[2]);
    G[15] = spu_ai(G[15], 32);
}

/* Guest func_000009A0: one dot-product-pass element. Cursors on entry:
 * G[12] input (+32/element), G[13] 16-byte out (+16), G[14] counter
 * (pre-increment value). */
static void j_dot_elem(spu_context *ctx, u128 *G)
{
    G[37] = spu_il(12);
    G[35] = spu_ls_read128(ctx, G[12]._u32[0] + 0x0);
    G[36] = spu_il(4);
    G[9]  = spu_ls_read128(ctx, G[12]._u32[0] + 0x10);
    G[26] = spu_ai(G[12], 12);
    G[25] = spu_ls_read128(ctx, G[12]._u32[0] + G[37]._u32[0]);
    G[74] = spu_il(8);
    G[67] = spu_ls_read128(ctx, G[12]._u32[0] + G[36]._u32[0]);
    G[73] = spu_il(20);
    G[72] = spu_fsmbi(0x0);
    G[70] = spu_il(60);
    G[64] = spu_ls_read128(ctx, G[12]._u32[0] + G[74]._u32[0]);
    G[33] = spu_ilhu(0x3F80);
    G[28] = spu_rotqby(G[35], G[12]);
    G[68] = spu_ai(G[12], 4);
    G[2]  = spu_ls_read128(ctx, G[12]._u32[0] + G[73]._u32[0]);
    G[11] = spu_ai(G[12], 8);
    G[71] = spu_rotqby(G[25], G[26]);
    G[3]  = spu_ai(G[12], 20);
    G[4]  = spu_ls_read128(ctx, G[23]._u32[0] + G[70]._u32[0]);
    G[10] = spu_ai(G[23], 60);
    G[66] = spu_shufb(G[28], G[33], G[21]);
    G[14] = spu_ai(G[14], 1);
    G[65] = spu_rotqby(G[9], G[12]);
    G[12] = spu_ai(G[12], 32);
    G[5]  = spu_shufb(G[71], G[72], G[21]);
    G[25] = spu_ceq(G[27], G[14]);
    G[8]  = spu_rotqby(G[67], G[68]);
    G[6]  = spu_rotqby(G[64], G[11]);
    G[7]  = spu_rotqby(G[2], G[3]);
    G[69] = spu_shufb(G[65], G[5], G[20]);
    G[63] = spu_shufb(G[8], G[66], G[20]);
    G[78] = spu_rotqby(G[4], G[10]);
    G[75] = spu_shufb(G[7], G[69], G[19]);
    G[76] = spu_shufb(G[6], G[63], G[19]);
    G[77] = spu_shufb(G[78], G[78], G[24]);
    G[79] = spu_fma(G[75], G[77], G[76]);
    G[40] = spu_shufb(G[79], G[79], G[22]);
    G[39] = spu_shufb(G[79], G[79], G[17]);
    G[35] = spu_shufb(G[79], G[79], G[15]);
    G[37] = spu_shufb(G[79], G[79], G[16]);
    G[38] = spu_fm(G[30], G[40]);
    G[36] = spu_fm(G[31], G[39]);
    G[28] = spu_fma(G[29], G[37], G[38]);
    G[33] = spu_fma(G[32], G[35], G[36]);
    G[26] = spu_fa(G[28], G[33]);
    spu_ls_write128(ctx, G[13]._u32[0] + 0x0, G[26]);
    G[13] = spu_ai(G[13], 16);
}

typedef struct {
    spu_context *ctx;
    const u128  *gstate;    /* per-group state after the pass setup */
} j_pass_arg;

static void j_xform_chunk(void *varg, uint32_t b, uint32_t e)
{
    const j_pass_arg *a = (const j_pass_arg *)varg;
    SPU_ALIGN16 u128 G[128];
    memcpy(G, a->gstate, sizeof G);
    for (int i = 0; i < 4; i++) {
        G[14]._u32[i] += 32u * b;
        G[16]._u32[i] += 16u * b;
        G[15]._u32[i] += 32u * b;
        G[17]._u32[i]  = b;
    }
    for (uint32_t k = b; k < e; k++)
        j_xform_elem(a->ctx, G);
}

static void j_dot_chunk(void *varg, uint32_t b, uint32_t e)
{
    const j_pass_arg *a = (const j_pass_arg *)varg;
    SPU_ALIGN16 u128 G[128];
    memcpy(G, a->gstate, sizeof G);
    for (int i = 0; i < 4; i++) {
        G[12]._u32[i] += 32u * b;
        G[13]._u32[i] += 16u * b;
        G[14]._u32[i]  = b;
    }
    for (uint32_t k = b; k < e; k++)
        j_dot_elem(a->ctx, G);
}

/* Guest func_000007DC (cull pass, index bound > 0): loop setup. After
 * this, G[12] = index cursor (group base), G[10] = record cursor,
 * G[22] = group base, G[13] = cumulative element base vector. */
static void j_cull_setup(spu_context *ctx, u128 *G)
{
    G[24] = spu_shli(G[45], 2);
    G[28] = spu_ls_read128(ctx, 0xB80);
    G[14] = spu_ilhu(0x3F80);
    G[31] = spu_ls_read128(ctx, 0xB90);
    G[25] = spu_ilhu(0x405);
    G[30] = spu_ls_read128(ctx, 0xBA0);
    G[12] = spu_ori(G[45], 0);
    G[29] = spu_ls_read128(ctx, 0xBB0);
    G[10] = spu_a(G[61], G[24]);
    G[22] = spu_shlqbyi(G[45], 0);
    G[24] = spu_ori(G[14], 1);
    G[13] = spu_shlqbyi(G[43], 0);
    G[25] = spu_iohl(G[25], 0x607);
    G[15] = spu_ila(0xFFFF);
    G[21] = spu_ila(0x10203);
    G[20] = spu_ilhu(0xBF80);
}

#define J_DEC_8E0 1   /* write the record via guest func_000008E0 */
#define J_DEC_B00 2   /* write the record via guest func_00000B00 */

/* Guest func_00000820 + the test portion of func_00000A70 for one
 * triangle: decides which write path the guest takes. Pure reads and
 * register math -- safe to run in parallel. On return, G[16], G[19],
 * G[17] hold the triangle's masked stream indices (the 0xB00 write
 * path's inputs); G[14] holds spu_il(4), as the guest's 0x8E0 write
 * path expects. */
static int j_cull_test(spu_context *ctx, u128 *G)
{
    G[36] = spu_sfh(G[22], G[12]);
    G[33] = spu_il(2);
    G[35] = spu_and(G[15], G[36]);
    G[14] = spu_il(4);
    G[9]  = spu_a(G[35], G[35]);
    G[26] = spu_a(G[34], G[9]);
    G[73] = spu_ls_read128(ctx, G[9]._u32[0] + G[34]._u32[0]);
    G[74] = spu_ai(G[26], 14);
    G[71] = spu_ls_read128(ctx, G[26]._u32[0] + G[33]._u32[0]);
    G[72] = spu_ai(G[26], 16);
    G[68] = spu_ls_read128(ctx, G[26]._u32[0] + G[14]._u32[0]);
    G[70] = spu_ai(G[26], 18);
    G[16] = spu_rotqby(G[73], G[74]);
    G[19] = spu_rotqby(G[71], G[72]);
    G[17] = spu_rotqby(G[68], G[70]);
    G[16] = spu_and(G[15], G[16]);
    G[19] = spu_and(G[15], G[19]);
    G[17] = spu_and(G[15], G[17]);
    G[11] = spu_shli(G[16], 4);
    G[67] = spu_shli(G[19], 4);
    G[8]  = spu_shli(G[17], 4);
    G[11] = spu_ls_read128(ctx, G[11]._u32[0] + G[18]._u32[0]);
    G[9]  = spu_ls_read128(ctx, G[67]._u32[0] + G[18]._u32[0]);
    G[8]  = spu_ls_read128(ctx, G[8]._u32[0] + G[18]._u32[0]);
    G[66] = spu_rotqbyi(G[11], 12);
    G[5]  = spu_rotqbyi(G[9], 12);
    G[65] = spu_rotqbyi(G[8], 12);
    G[2]  = spu_shufb(G[66], G[66], G[21]);
    G[3]  = spu_shufb(G[5], G[5], G[21]);
    G[64] = spu_shufb(G[65], G[65], G[21]);
    G[4]  = spu_fm(G[2], G[20]);
    G[6]  = spu_fm(G[3], G[20]);
    G[77] = spu_fm(G[64], G[20]);
    G[69] = spu_fcgt(G[11], G[2]);
    G[63] = spu_fcgt(G[9], G[3]);
    G[76] = spu_fcgt(G[8], G[64]);
    G[78] = spu_fcgt(G[4], G[11]);
    G[7]  = spu_fcgt(G[6], G[9]);
    G[75] = spu_and(G[69], G[63]);
    G[40] = spu_and(G[78], G[7]);
    G[79] = spu_fcgt(G[77], G[8]);
    G[38] = spu_and(G[75], G[76]);
    G[39] = spu_and(G[40], G[79]);
    G[32] = spu_or(G[38], G[39]);
    G[45] = spu_orx(G[32]);
    if (G[45]._u32[0] != 0) return J_DEC_8E0;
    /* func_00000A70: w-sign pre-test, then the divide-based test */
    G[3]  = spu_ilhu(0xC0D);
    G[38] = spu_shufb(G[9], G[8], G[28]);
    G[26] = spu_il(0);
    G[3]  = spu_iohl(G[3], 0xE0F);
    G[7]  = spu_shufb(G[11], G[11], G[3]);
    G[32] = spu_fm(G[7], G[38]);
    G[45] = spu_fcgt(G[26], G[32]);
    G[37] = spu_orx(G[45]);
    if (G[37]._u32[0] != 0) return J_DEC_B00;
    G[70] = spu_shufb(G[8], G[8], G[3]);
    G[68] = spu_shufb(G[9], G[9], G[3]);
    G[74] = spu_frest(G[7]);
    G[73] = spu_frest(G[70]);
    G[72] = spu_frest(G[68]);
    G[71] = spu_fi(G[7], G[74]);
    G[5]  = spu_fi(G[70], G[73]);
    G[67] = spu_fi(G[68], G[72]);
    G[66] = spu_fnms(G[7], G[71], G[24]);
    G[4]  = spu_fm(G[11], G[71]);
    G[6]  = spu_fnms(G[70], G[5], G[24]);
    G[2]  = spu_fnms(G[68], G[67], G[24]);
    G[3]  = spu_fm(G[9], G[67]);
    G[65] = spu_fm(G[8], G[5]);
    G[64] = spu_fma(G[66], G[4], G[4]);
    G[69] = spu_fma(G[2], G[3], G[3]);
    G[63] = spu_fma(G[6], G[65], G[65]);
    G[7]  = spu_shufb(G[64], G[64], G[30]);
    G[78] = spu_shufb(G[69], G[63], G[31]);
    G[76] = spu_fs(G[78], G[7]);
    G[77] = spu_shufb(G[76], G[76], G[29]);
    G[79] = spu_fm(G[76], G[77]);
    G[75] = spu_shufb(G[79], G[79], G[25]);
    G[40] = spu_fs(G[79], G[75]);
    G[39] = spu_fcgt(G[26], G[40]);
    if (G[39]._u32[0] != 0) return J_DEC_8E0;
    return J_DEC_B00;
}

/* Guest func_000008E0: merge the base vector G[13] into the 12-byte
 * record at G[10]. Expects G[14] == spu_il(4) (set by func_00000820). */
static void j_write_8e0(spu_context *ctx, u128 *G)
{
    G[8]  = spu_il(8);
    G[9]  = spu_ls_read128(ctx, G[10]._u32[0] + 0x0);
    G[35] = spu_cwd(G[10], 0);
    G[26] = spu_cwd(G[10], 4);
    G[17] = spu_cwd(G[10], 8);
    G[33] = spu_shufb(G[13], G[9], G[35]);
    spu_ls_write128(ctx, G[10]._u32[0] + 0x0, G[33]);
    G[16] = spu_ls_read128(ctx, G[10]._u32[0] + G[14]._u32[0]);
    G[19] = spu_shufb(G[13], G[16], G[26]);
    spu_ls_write128(ctx, G[10]._u32[0] + G[14]._u32[0], G[19]);
    G[11] = spu_ls_read128(ctx, G[10]._u32[0] + G[8]._u32[0]);
    G[14] = spu_shufb(G[13], G[11], G[17]);
    spu_ls_write128(ctx, G[10]._u32[0] + G[8]._u32[0], G[14]);
}

/* Guest func_00000B00: merge (cumulative base + stream index) values
 * into the 12-byte record at G[10]. Sets G[13] = base + third index,
 * which the guest carries into later func_000008E0 writes. */
static void j_write_b00(spu_context *ctx, u128 *G)
{
    G[78] = spu_a(G[43], G[16]);
    G[77] = spu_ls_read128(ctx, G[10]._u32[0] + 0x0);
    G[38] = spu_il(4);
    G[7]  = spu_cwd(G[10], 0);
    G[79] = spu_a(G[43], G[19]);
    G[36] = spu_il(8);
    G[75] = spu_cwd(G[10], 4);
    G[13] = spu_a(G[43], G[17]);
    G[32] = spu_cwd(G[10], 8);
    G[76] = spu_shufb(G[78], G[77], G[7]);
    spu_ls_write128(ctx, G[10]._u32[0] + 0x0, G[76]);
    G[40] = spu_ls_read128(ctx, G[10]._u32[0] + G[38]._u32[0]);
    G[39] = spu_shufb(G[79], G[40], G[75]);
    spu_ls_write128(ctx, G[10]._u32[0] + G[38]._u32[0], G[39]);
    G[45] = spu_ls_read128(ctx, G[10]._u32[0] + G[36]._u32[0]);
    G[37] = spu_shufb(G[13], G[45], G[32]);
    spu_ls_write128(ctx, G[10]._u32[0] + G[36]._u32[0], G[37]);
}

/* The cull loop transcribed directly (guest funcs 0x820..0x940 with
 * the writes inline), used for small triangle counts and whenever the
 * index bound is too large for the precomputed-count parallel path.
 * G is the post-setup state; runs to completion of the group. */
static void j_cull_serial(spu_context *ctx, u128 *G)
{
    for (;;) {
        int dec = j_cull_test(ctx, G);
        if (dec == J_DEC_8E0) j_write_8e0(ctx, G);
        else                 j_write_b00(ctx, G);
        /* func_00000918: advance, loop while bound > consumed */
        G[12] = spu_ai(G[12], 3);
        G[10] = spu_ai(G[10], 12);
        G[64] = spu_sfh(G[22], G[12]);
        G[63] = spu_and(G[15], G[64]);
        G[69] = spu_cgt(G[41], G[63]);
        if (G[69]._u32[0] == 0) break;
    }
    G[45] = spu_ori(G[12], 0);
}

typedef struct {
    spu_context *ctx;
    const u128  *gstate;    /* state after j_cull_setup */
    uint8_t     *decision;  /* per triangle: J_DEC_8E0 / J_DEC_B00 */
    u128        *stash;     /* per triangle: {G[16], G[19], G[17]} */
} j_cull_arg;

static void j_cull_chunk(void *varg, uint32_t b, uint32_t e)
{
    const j_cull_arg *a = (const j_cull_arg *)varg;
    SPU_ALIGN16 u128 G[128];
    for (uint32_t t = b; t < e; t++) {
        memcpy(G, a->gstate, sizeof G);
        for (int i = 0; i < 4; i++) {
            G[12]._u32[i] += 3u * t;
            G[10]._u32[i] += 12u * t;
        }
        a->decision[t] = (uint8_t)j_cull_test(a->ctx, G);
        a->stash[3 * t + 0] = G[16];
        a->stash[3 * t + 1] = G[19];
        a->stash[3 * t + 2] = G[17];
    }
}

/* Parallel cull: tests across the pool, then the writes serially in
 * triangle order, reproducing the guest's gpr[13] carry exactly. */
static void j_cull_parallel(spu_context *ctx, const u128 *gstate, uint32_t tris)
{
    uint8_t *decision = (uint8_t *)malloc(tris);
    u128 *stash = (u128 *)malloc((size_t)tris * 3 * sizeof(u128));
    if (!decision || !stash) {
        free(decision);
        free(stash);
        SPU_ALIGN16 u128 G[128];
        memcpy(G, gstate, sizeof G);
        j_cull_serial(ctx, G);
        return;
    }
    j_cull_arg a;
    a.ctx = ctx; a.gstate = gstate; a.decision = decision; a.stash = stash;
    j_pool_run(j_cull_chunk, &a, tris);

    SPU_ALIGN16 u128 W[128];
    u128 carried = gstate[13];
    for (uint32_t t = 0; t < tris; t++) {
        memcpy(W, gstate, sizeof W);
        for (int i = 0; i < 4; i++)
            W[10]._u32[i] += 12u * t;
        W[14] = spu_il(4);
        if (decision[t] == J_DEC_8E0) {
            W[13] = carried;
            j_write_8e0(ctx, W);
        } else {
            W[16] = stash[3 * t + 0];
            W[19] = stash[3 * t + 1];
            W[17] = stash[3 * t + 2];
            j_write_b00(ctx, W);
            carried = spu_a(gstate[43], stash[3 * t + 2]);
        }
    }
    free(decision);
    free(stash);
}

/* Native replacement for spu_job_00187280_spu_func_000004A0. Reads the
 * argument registers from ctx, computes into guest LS, and leaves
 * ctx->gpr exactly as it found it (see the contract at the top). */
void ps3_job187280_compute_native(spu_context *ctx)
{
    SPU_ALIGN16 u128 base[128];
    memcpy(base, ctx->gpr, sizeof base);
    if (!j_prologue(ctx, base))
        return;

    const int32_t  groups         = (int32_t)base[54]._u32[0];
    const int      mode_transform = base[55]._u32[0] != 0;
    const int      have_elems     = base[56]._u32[0] != 0;
    const int      cull_on        = base[57]._u32[0] != 0;
    const int32_t  r7             = (int32_t)base[41]._u32[0];
    const uint32_t elems          = base[27]._u32[0];
    /* Triangles per group: the guest loop advances the index cursor by
     * 3 per triangle and stops once the consumed count reaches the
     * bound, so the count is data-independent. */
    const uint32_t tris     = cull_on ? (uint32_t)(((int64_t)r7 + 2) / 3) : 0;
    const uint32_t consumed = 3u * tris;

    for (int32_t g = 0; g < groups; g++) {
        SPU_ALIGN16 u128 G[128];
        memcpy(G, base, sizeof G);
        /* Carried-register state at this group's start (the advances
         * the guest applies in func_00000940 / func_00000670's exit /
         * func_00000918's exit, per group). */
        for (int i = 0; i < 4; i++) {
            G[23]._u32[i] += 64u * (uint32_t)g;
            G[44]._u32[i] += 16u * (uint32_t)g;
            G[43]._u32[i] += (uint32_t)g * base[27]._u32[i];
            G[53]._u32[i]  = (uint32_t)g * base[27]._u32[i];
            G[45]._u32[i]  = consumed * (uint32_t)g;
        }
        /* func_0000052C: gated +0x30 quadword copy */
        if (G[58]._u32[0] != 0) {
            G[7] = spu_ls_read128(ctx, G[23]._u32[0] + 0x30);
            spu_ls_write128(ctx, G[44]._u32[0] + 0x0, G[7]);
        }
        j_compose(ctx, G);
        if (!mode_transform) {
            if (have_elems) {
                j_pass_arg a;
                j_dot_setup(G);
                a.ctx = ctx; a.gstate = G;
                j_pool_run(j_dot_chunk, &a, elems);
            }
        } else if (have_elems) {
            j_pass_arg a;
            j_xform_setup(G);
            a.ctx = ctx; a.gstate = G;
            j_pool_run(j_xform_chunk, &a, elems);
        }
        if (cull_on) {
            j_cull_setup(ctx, G);
            if (tris >= 48 && r7 <= 30000)
                j_cull_parallel(ctx, G, tris);
            else
                j_cull_serial(ctx, G);
        }
    }
}

