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
#include <stdatomic.h>

/* POOL SPIN EXPERIMENT: spin-then-park waits. Workers that find no
 * work spin briefly watching the submission sequence before parking
 * on the cond; the submitting caller spins briefly on its own job's
 * atomic completion counters before waiting on the finished cond.
 * The mutex/cond protocol is unchanged and remains the authority:
 * every spin observation is only a hint, re-verified under the mutex
 * before any state is touched (and the job fields a worker could
 * race with are never dereferenced outside the mutex -- workers spin
 * on the global sequence counter only). */
#ifndef J_SPIN_WORKER
#define J_SPIN_WORKER 256
#endif
#ifndef J_SPIN_CALLER
#define J_SPIN_CALLER 1000
#endif
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#define J_PAUSE() _mm_pause()
#else
#define J_PAUSE() ((void)0)
#endif

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
    _Atomic uint32_t a_done;   /* mirrors of done/active, written under */
    _Atomic uint32_t a_active; /* the mutex, read lock-free ONLY by the */
                               /* job's owner for its spin wait        */
    struct jjob   *qnext;
    pthread_cond_t finished;   /* broadcast when done==n_chunks && active==0 */
} jjob_t;

static pthread_mutex_t j_pool_mtx  = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  j_pool_work = PTHREAD_COND_INITIALIZER;
static _Atomic(jjob_t *) j_pool_head = 0;
static jjob_t         *j_pool_tail = 0;
static _Atomic uint32_t j_pool_seq = 0;  /* bumped on every enqueue */
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
        atomic_store_explicit(&j->a_done, j->done, memory_order_release);
        pthread_mutex_unlock(&j_pool_mtx);
    }
}

static int j_spin_worker = J_SPIN_WORKER;

static void *j_worker(void *unused)
{
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&j_pool_mtx);
        jjob_t *j = atomic_load_explicit(&j_pool_head, memory_order_relaxed);
        if (j && j->done != j->n_chunks) {
            /* Help with the head job. A worker must never re-take a
             * completed job: its chunks are gone, so draining it is a
             * no-op that would spin against the owner's wait and
             * starve the pool mutex (found by the harness, which
             * livelocked at full CPU). */
            j->active++;
            atomic_store_explicit(&j->a_active, j->active,
                                  memory_order_release);
            pthread_mutex_unlock(&j_pool_mtx);
            j_drain(j);
            pthread_mutex_lock(&j_pool_mtx);
            j->active--;
            atomic_store_explicit(&j->a_active, j->active,
                                  memory_order_release);
            if (j->done == j->n_chunks && j->active == 0)
                pthread_cond_broadcast(&j->finished);
            pthread_mutex_unlock(&j_pool_mtx);
            continue;
        }
        /* Nothing to help with: no job queued, or the head job is
         * complete and its owner is about to unlink it (the owner
         * broadcasts j_pool_work when it does). Spin briefly watching
         * the submission sequence -- in this workload the next group's
         * job is typically submitted within microseconds, and a cond
         * wake costs far more than the spin -- then park. */
        uint32_t seq = atomic_load_explicit(&j_pool_seq,
                                            memory_order_acquire);
        pthread_mutex_unlock(&j_pool_mtx);
        int bumped = 0;
        for (int s = 0; s < j_spin_worker; s++) {
            if (atomic_load_explicit(&j_pool_seq, memory_order_acquire)
                    != seq) {
                bumped = 1;
                break;
            }
            J_PAUSE();
        }
        if (bumped)
            continue;
        pthread_mutex_lock(&j_pool_mtx);
        j = atomic_load_explicit(&j_pool_head, memory_order_relaxed);
        if (!j || j->done == j->n_chunks)
            pthread_cond_wait(&j_pool_work, &j_pool_mtx);
        pthread_mutex_unlock(&j_pool_mtx);
    }
    return 0;
}

static void j_pool_init(void)
{
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    int want = (n > 1) ? (int)n - 1 : 0;
    if (want > 15) want = 15;
    /* J187280_WORKERS / J187280_SPIN: pool size and how long an idle worker
     * spins (pause iterations) before parking. Tuning experiments. */
    /* Defaults: 4 workers that spin ~200 us before parking. The job is
     * called every ~250 us with ~13 us groups, so parked workers (a futex
     * wake costs tens of us) arrived after the caller had done the work
     * itself; 15 workers also meant 15 threads waking per submission. */
    if (want > 4) want = 4;
    j_spin_worker = 20000;
    const char *w = getenv("J187280_WORKERS");
    if (w && atoi(w) > 0) { long mx = (n > 1) ? n - 1 : 0; want = atoi(w) < mx ? atoi(w) : (int)mx; }
    const char *sp = getenv("J187280_SPIN");
    if (sp && atoi(sp) > 0) j_spin_worker = atoi(sp);
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
static void j_pool_run_ex(jchunk_fn fn, void *arg, uint32_t n,
                          uint32_t min_par, uint32_t per_worker)
{
    if (n == 0)
        return;
    static pthread_once_t once = PTHREAD_ONCE_INIT;
    pthread_once(&once, j_pool_init);
    if (!j_pool_ready || n < min_par) {
        fn(arg, 0, n);
        return;
    }
    uint32_t target = (uint32_t)(j_pool_nworkers + 1) * per_worker;
    uint32_t chunk = (n + target - 1) / target;
    if (chunk == 0) chunk = 1;

    jjob_t j;
    memset(&j, 0, sizeof j);
    j.fn = fn; j.arg = arg; j.n_items = n; j.chunk = chunk;
    j.n_chunks = (n + chunk - 1) / chunk;
    pthread_cond_init(&j.finished, 0);

    pthread_mutex_lock(&j_pool_mtx);
    if (j_pool_tail) j_pool_tail->qnext = &j;
    else atomic_store_explicit(&j_pool_head, &j, memory_order_release);
    j_pool_tail = &j;
    atomic_fetch_add_explicit(&j_pool_seq, 1, memory_order_release);
    pthread_cond_broadcast(&j_pool_work);
    pthread_mutex_unlock(&j_pool_mtx);

    j_drain(&j);

    /* Spin briefly on this job's own atomic counters: the tail chunks
     * usually complete within microseconds, and entering the cond
     * wait costs a futex round-trip even when the broadcast has
     * already happened. The mutex wait below re-verifies the plain
     * counters and remains the authority for the unlink. */
    for (int s = 0; s < J_SPIN_CALLER; s++) {
        if (atomic_load_explicit(&j.a_done, memory_order_acquire)
                == j.n_chunks &&
            atomic_load_explicit(&j.a_active, memory_order_acquire) == 0)
            break;
        J_PAUSE();
    }
    pthread_mutex_lock(&j_pool_mtx);
    while (!(j.done == j.n_chunks && j.active == 0))
        pthread_cond_wait(&j.finished, &j_pool_mtx);
    /* unlink (only safe now: no worker can touch the job any more) */
    jjob_t *head = atomic_load_explicit(&j_pool_head, memory_order_relaxed);
    if (head == &j) {
        atomic_store_explicit(&j_pool_head, j.qnext, memory_order_relaxed);
    } else {
        jjob_t *q = head;
        while (q && q->qnext != &j) q = q->qnext;
        if (q) q->qnext = j.qnext;
    }
    if (j_pool_tail == &j) {
        j_pool_tail = 0;
        for (jjob_t *q = atomic_load_explicit(&j_pool_head,
                                              memory_order_relaxed);
             q; q = q->qnext)
            j_pool_tail = q;
    }
    /* wake workers parked on the completed head so they move on to
     * whatever is queued behind it */
    pthread_cond_broadcast(&j_pool_work);
    pthread_mutex_unlock(&j_pool_mtx);
    pthread_cond_destroy(&j.finished);
}

static void j_pool_run(jchunk_fn fn, void *arg, uint32_t n)
{
    j_pool_run_ex(fn, arg, n, 256, 2);
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
/* priv: when non-null, the 16-byte scratch result (normally stored at the
 * shared LS cursor G[16]) goes here instead -- the two-phase group path. */
static inline void j_xform_elem_p(spu_context *ctx, u128 *G, u128 *priv)
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
    if (priv) *priv = G[73];
    else spu_ls_write128(ctx, G[16]._u32[0] + 0x0, G[73]);
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

static void j_xform_elem(spu_context *ctx, u128 *G) { j_xform_elem_p(ctx, G, 0); }

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
    j_pool_run_ex(j_cull_chunk, &a, tris, 256, 1);

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
/* One call's shared inputs, for running its groups (in parallel or not). */
typedef struct {
    spu_context *ctx;
    const u128 *base;
    int32_t groups, r7;
    int mode_transform, have_elems, cull_on;
    uint32_t elems, tris, consumed;
    int inner_par;     /* may a group use the pool itself (only when groups run serially) */
    /* LS pages the pool's workers may have written (their ls_dirty updates
     * race), and whether that set is unknown (claim the whole store). */
    uint64_t dirty_extra;
    int claim_all;
} j_job_arg;

/* The ls_dirty bits for [lsa, lsa + size): spu_ls_mark's arithmetic. */
static uint64_t j_pages(uint32_t lsa, uint32_t size)
{
    if (!size) return 0;
    lsa &= SPU_LS_MASK;
    uint32_t last = lsa + size - 1u;
    if (last > SPU_LS_MASK) return ~0ull;   /* wraps: be conservative */
    const uint32_t p0 = lsa >> 12, p1 = last >> 12;
    return (p1 - p0 >= 63u) ? ~0ull : (((2ull << (p1 - p0)) - 1ull) << p0);
}

/* Groups [b, e) of one call: the guest's per-group loop body, unchanged. */
static void j_groups_chunk(void *varg, uint32_t b, uint32_t e)
{
    const j_job_arg *ja = (const j_job_arg *)varg;
    spu_context *ctx = ja->ctx;
    const u128 *base = ja->base;
    for (uint32_t g = b; g < e; g++) {
        SPU_ALIGN16 u128 G[128];
        memcpy(G, base, sizeof G);
        /* Carried-register state at this group's start (the advances
         * the guest applies in func_00000940 / func_00000670's exit /
         * func_00000918's exit, per group). */
        for (int i = 0; i < 4; i++) {
            G[23]._u32[i] += 64u * g;
            G[44]._u32[i] += 16u * g;
            G[43]._u32[i] += g * base[27]._u32[i];
            G[53]._u32[i]  = g * base[27]._u32[i];
            G[45]._u32[i]  = ja->consumed * g;
        }
        /* func_0000052C: gated +0x30 quadword copy */
        if (G[58]._u32[0] != 0) {
            G[7] = spu_ls_read128(ctx, G[23]._u32[0] + 0x30);
            spu_ls_write128(ctx, G[44]._u32[0] + 0x0, G[7]);
        }
        j_compose(ctx, G);
        if (!ja->mode_transform) {
            if (ja->have_elems) {
                j_pass_arg a;
                j_dot_setup(G);
                a.ctx = ctx; a.gstate = G;
                if (ja->inner_par) j_pool_run_ex(j_dot_chunk, &a, ja->elems, 128, 1);
                else j_dot_chunk(&a, 0, ja->elems);
            }
        } else if (ja->have_elems) {
            j_pass_arg a;
            j_xform_setup(G);
            a.ctx = ctx; a.gstate = G;
            if (ja->inner_par) j_pool_run_ex(j_xform_chunk, &a, ja->elems, 128, 1);
            else j_xform_chunk(&a, 0, ja->elems);
        }
        if (ja->cull_on) {
            j_cull_setup(ctx, G);
            if (ja->inner_par && ja->tris >= 48 && ja->r7 <= 30000)
                j_cull_parallel(ctx, G, ja->tris);
            else
                j_cull_serial(ctx, G);
        }
    }
}

/* Two-phase groups (transform mode). Phase 1, across the pool: each group's
 * 0x52C copy, compose and transform -- they read shared input and write only
 * that group's own LS blocks -- with the transform's 16-byte-per-element
 * scratch (the shared LS area at G[16], which every group overwrites) kept
 * in a private buffer, plus the group's register state for the cull. Phase
 * 2, in group order: copy the group's scratch into place, then its cull,
 * which appends records after the previous group's. The local store ends
 * exactly as the serial loop leaves it (J187280_VERIFY checks every byte). */
typedef struct {
    j_job_arg *ja;
    u128 *gsnap;       /* groups x 128 registers after the pass setup */
    u128 *scratch;     /* groups x elems scratch results */
} j_tp_arg;

static void j_tp_phase1(void *varg, uint32_t b, uint32_t e)
{
    const j_tp_arg *t = (const j_tp_arg *)varg;
    const j_job_arg *ja = t->ja;
    spu_context *ctx = ja->ctx;
    const u128 *base = ja->base;
    for (uint32_t g = b; g < e; g++) {
        u128 *G = t->gsnap + (size_t)g * 128;
        memcpy(G, base, 128 * sizeof(u128));
        for (int i = 0; i < 4; i++) {
            G[23]._u32[i] += 64u * g;
            G[44]._u32[i] += 16u * g;
            G[43]._u32[i] += g * base[27]._u32[i];
            G[53]._u32[i]  = g * base[27]._u32[i];
            G[45]._u32[i]  = ja->consumed * g;
        }
        if (G[58]._u32[0] != 0) {
            G[7] = spu_ls_read128(ctx, G[23]._u32[0] + 0x30);
            spu_ls_write128(ctx, G[44]._u32[0] + 0x0, G[7]);
        }
        j_compose(ctx, G);
        if (ja->have_elems) {
            j_xform_setup(G);
            SPU_ALIGN16 u128 X[128];
            memcpy(X, G, sizeof X);
            for (int i = 0; i < 4; i++) X[17]._u32[i] = 0;
            u128 *sc = t->scratch + (size_t)g * ja->elems;
            for (uint32_t k = 0; k < ja->elems; k++)
                j_xform_elem_p(ctx, X, &sc[k]);
        }
    }
}

/* 1 when done, 0 when the call is not suitable (caller runs it serially). */
static int j_groups_two_phase(j_job_arg *ja)
{
    if (!ja->mode_transform || ja->groups < 2) return 0;
    const size_t ng = (size_t)ja->groups;
    const size_t scratch_bytes = ja->have_elems ? (size_t)ja->elems * 16u : 0;
    j_tp_arg t;
    t.ja = ja;
    t.gsnap = (u128 *)aligned_alloc(16, ng * 128 * sizeof(u128));
    t.scratch = scratch_bytes ? (u128 *)aligned_alloc(16, ng * scratch_bytes) : 0;
    if (!t.gsnap || (scratch_bytes && !t.scratch)) { free(t.gsnap); free(t.scratch); return 0; }
    j_pool_run_ex(j_tp_phase1, &t, (uint32_t)ng, 2, 1);
    spu_context *ctx = ja->ctx;
    /* Phase 1's LS writes (by workers, racing on ls_dirty): each group's
     * transform output (32 bytes per element from its G[15]) and its 0x52C
     * quadword copy (16 bytes at G[44]). Everything else is written below,
     * on this thread, and marks itself. */
    for (size_t g = 0; g < ng; g++) {
        const u128 *G = t.gsnap + g * 128;
        if (ja->have_elems) ja->dirty_extra |= j_pages(G[15]._u32[0], 32u * ja->elems);
        if (G[58]._u32[0] != 0) ja->dirty_extra |= j_pages(G[44]._u32[0], 16u);
    }
    if (ja->cull_on && ja->tris >= 48 && ja->r7 <= 30000) ja->claim_all = 1;   /* parallel cull below */
    for (size_t g = 0; g < ng; g++) {
        u128 *G = t.gsnap + g * 128;
        if (scratch_bytes) {
            /* the transform's scratch, exactly where the serial loop leaves it */
            const uint32_t dst = G[16]._u32[0] & SPU_LS_MASK;
            const u128 *sc = t.scratch + g * ja->elems;
            if (dst + scratch_bytes <= SPU_LS_SIZE) {
                for (uint32_t k = 0; k < ja->elems; k++) spu_ls_write128(ctx, dst + 16u * k, sc[k]);
            } else {
                for (uint32_t k = 0; k < ja->elems; k++) spu_ls_write128(ctx, (dst + 16u * k) & SPU_LS_MASK, sc[k]);
            }
        }
        if (ja->cull_on) {
            j_cull_setup(ctx, G);
            if (ja->tris >= 48 && ja->r7 <= 30000)
                j_cull_parallel(ctx, G, ja->tris);
            else
                j_cull_serial(ctx, G);
        }
    }
    free(t.gsnap);
    free(t.scratch);
    return 1;
}

/* J187280_VERIFY=1: run the call serially (the reference), then again with
 * its groups in parallel from the same starting LS, and compare every byte. */
static void j_verify_groups(j_job_arg *ja)
{
    static _Atomic uint64_t vcalls, vbad;
    uint8_t *pre = (uint8_t *)malloc(SPU_LS_SIZE), *ref = (uint8_t *)malloc(SPU_LS_SIZE);
    if (!pre || !ref) {
        free(pre); free(ref);
        ja->inner_par = 1;
        j_groups_chunk(ja, 0, (uint32_t)ja->groups);
        return;
    }
    memcpy(pre, ja->ctx->ls, SPU_LS_SIZE);
    ja->inner_par = 1;
    j_groups_chunk(ja, 0, (uint32_t)ja->groups);
    memcpy(ref, ja->ctx->ls, SPU_LS_SIZE);
    memcpy(ja->ctx->ls, pre, SPU_LS_SIZE);
    ja->inner_par = 0;
    if (!j_groups_two_phase(ja))
        j_groups_chunk(ja, 0, (uint32_t)ja->groups);
    const int same = memcmp(ref, ja->ctx->ls, SPU_LS_SIZE) == 0;
    const uint64_t n = atomic_fetch_add(&vcalls, 1) + 1;
    if (!same) {
        const uint64_t bad = atomic_fetch_add(&vbad, 1) + 1;
        if (bad <= 5) {
            uint32_t first = 0;
            while (first < SPU_LS_SIZE && ref[first] == ja->ctx->ls[first]) first++;
            fprintf(stderr, "[J187280 VERIFY] MISMATCH call %llu: groups=%d elems=%u tris=%u first diff at LS 0x%05X\n",
                    (unsigned long long)n, ja->groups, ja->elems, ja->tris, first);
        }
        memcpy(ja->ctx->ls, ref, SPU_LS_SIZE);   /* keep the reference result */
    }
    if (n % 20000 == 0)
        fprintf(stderr, "[J187280 VERIFY] %llu calls compared, %llu mismatches\n",
                (unsigned long long)n, (unsigned long long)atomic_load(&vbad));
    free(pre); free(ref);
}

/* J187280_STATS=1: once a second, log how this job is being called -- calls,
 * groups, elements per group (how many reached the pool's parallel
 * threshold), cull triangles, distinct calling threads, time inside. */
#include <stdio.h>
#include <stdatomic.h>
#include <time.h>
static int js_on = -1;
static _Atomic uint64_t js_calls, js_groups, js_elems, js_small, js_par, js_tris, js_ns, js_t0;
static _Atomic uint32_t js_tid_mask;
static uint64_t js_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ull + (uint64_t)ts.tv_nsec;
}
static void js_report(uint64_t now)
{
    uint64_t t0 = atomic_load(&js_t0);
    if (!t0) { atomic_compare_exchange_strong(&js_t0, &t0, now); return; }
    if (now - t0 < 1000000000ull) return;
    if (!atomic_compare_exchange_strong(&js_t0, &t0, now)) return;
    const uint64_t calls = atomic_exchange(&js_calls, 0), groups = atomic_exchange(&js_groups, 0);
    const uint64_t elems = atomic_exchange(&js_elems, 0), small = atomic_exchange(&js_small, 0);
    const uint64_t par = atomic_exchange(&js_par, 0), tris = atomic_exchange(&js_tris, 0);
    const uint64_t ns = atomic_exchange(&js_ns, 0);
    const uint32_t mask = atomic_exchange(&js_tid_mask, 0);
    fprintf(stderr, "[J187280] /s: calls=%llu groups=%llu elems=%llu (avg %.1f/group; %llu groups <128 serial, %llu pooled) tris=%llu busy=%.1f%% threads~%d\n",
            (unsigned long long)calls, (unsigned long long)groups, (unsigned long long)elems,
            groups ? (double)elems / (double)groups : 0.0, (unsigned long long)small,
            (unsigned long long)par, (unsigned long long)tris, (double)ns / 1e7,
            __builtin_popcount(mask));
}

void ps3_job187280_compute_native(spu_context *ctx)
{
    if (js_on < 0) { const char* e = getenv("J187280_STATS"); js_on = e && atoi(e) == 1; }
    uint64_t js_start = js_on ? js_now() : 0;
    SPU_ALIGN16 u128 base[128];
    memcpy(base, ctx->gpr, sizeof base);
    if (!j_prologue(ctx, base))
        return;
    if (js_on) {
        const uint32_t e = base[27]._u32[0];
        const uint64_t g = (uint64_t)(int32_t)base[54]._u32[0];
        atomic_fetch_add(&js_calls, 1);
        atomic_fetch_add(&js_groups, g);
        atomic_fetch_add(&js_elems, g * e);
        if (e < 128) atomic_fetch_add(&js_small, g); else atomic_fetch_add(&js_par, g);
        if (base[57]._u32[0]) atomic_fetch_add(&js_tris, g * (uint64_t)(((int64_t)(int32_t)base[41]._u32[0] + 2) / 3));
        atomic_fetch_or(&js_tid_mask, 1u << (((uintptr_t)&base >> 12) % 32));
    }

    j_job_arg ja;
    ja.ctx = ctx;
    ja.base = base;
    ja.groups         = (int32_t)base[54]._u32[0];
    ja.mode_transform = base[55]._u32[0] != 0;
    ja.have_elems     = base[56]._u32[0] != 0;
    ja.cull_on        = base[57]._u32[0] != 0;
    ja.r7             = (int32_t)base[41]._u32[0];
    ja.elems          = base[27]._u32[0];
    /* Triangles per group: the guest loop advances the index cursor by
     * 3 per triangle and stops once the consumed count reaches the
     * bound, so the count is data-independent. */
    ja.tris     = ja.cull_on ? (uint32_t)(((int64_t)ja.r7 + 2) / 3) : 0;
    ja.consumed = 3u * ja.tris;
    ja.dirty_extra = 0;
    ja.claim_all = 0;

    /* Groups are independent (each one's state is derived from its index
     * and it writes its own LS ranges), so a call's groups run across the
     * pool. A group is ~80 elements, far below the per-group pool
     * threshold, so before this almost every call ran serially on one
     * thread -- but see below: it is off by default (J187280_GROUP_PAR=1 turns it on);
     * J187280_VERIFY=1 runs both and compares the whole local store. */
    static int jp_on = -1, jv_on = -1, tp_on = -1;
    /* The exact two-phase group path (see j_groups_two_phase): on by default,
     * J187280_TWOPHASE=0 restores the serial per-group loop. Verified byte-
     * identical over 60,000+ heavy-battle calls; ~35% less time per call. */
    if (tp_on < 0) { const char* e = getenv("J187280_TWOPHASE"); tp_on = !(e && *e && atoi(e) == 0); }
    /* OFF by default: groups share an LS scratch area (0x8400..) and append
     * their records one after another, so running them in parallel changes
     * the output (J187280_VERIFY found ~75% of calls differ). Experiment only. */
    if (jp_on < 0) { const char* e = getenv("J187280_GROUP_PAR"); jp_on = e && atoi(e) == 1; }
    if (jv_on < 0) { const char* e = getenv("J187280_VERIFY"); jv_on = e && atoi(e) == 1; }
    const int par = jp_on && ja.groups >= 2 &&
                    (uint64_t)ja.groups * (ja.elems + ja.tris) >= 192;
    static int jm_left = -1;   /* J187280_MAPW=N: map the LS bytes each group writes, for N calls */
    if (jm_left < 0) { const char* e = getenv("J187280_MAPW"); jm_left = e ? atoi(e) : 0; }
    if (jm_left > 0 && ja.groups >= 2) {
        jm_left--;
        uint8_t *before = (uint8_t *)malloc(SPU_LS_SIZE);
        ja.inner_par = 1;
        fprintf(stderr, "[J187280 MAPW] call: groups=%d elems=%u tris=%u\n", ja.groups, ja.elems, ja.tris);
        for (int32_t g = 0; g < ja.groups; g++) {
            memcpy(before, ctx->ls, SPU_LS_SIZE);
            j_groups_chunk(&ja, (uint32_t)g, (uint32_t)g + 1);
            /* print the changed ranges (merged when within 64 bytes) */
            char line[2048]; int len = snprintf(line, sizeof line, "  g%d:", g);
            int32_t start = -1, last = -1;
            for (int32_t i = 0; i <= SPU_LS_SIZE; i++) {
                const int diff = i < SPU_LS_SIZE && before[i] != ctx->ls[i];
                if (diff) { if (start < 0) start = i; last = i; }
                else if (start >= 0 && (i == SPU_LS_SIZE || i - last > 64)) {
                    if (len < (int)sizeof line - 32) len += snprintf(line + len, sizeof line - len, " %05X-%05X", start, last);
                    start = -1;
                }
            }
            fprintf(stderr, "%s\n", line);
        }
        free(before);
        ctx->ls_dirty = ~0ull;
        return;
    }
    const int tp = tp_on && ja.groups >= 2 && ja.mode_transform && ja.have_elems;
    if (jv_on && tp) {
        ja.claim_all = 1;
        j_verify_groups(&ja);
    } else if (tp && j_groups_two_phase(&ja)) {
        /* done */
    } else if (par) {
        ja.claim_all = 1;
        ja.inner_par = 0;
        j_pool_run_ex(j_groups_chunk, &ja, (uint32_t)ja.groups, 2, 1);
    } else {
        ja.inner_par = 1;
        /* A group's own pool use (big transforms, the parallel cull) writes
         * from workers: claim the store then. A serial call marks itself. */
        if ((ja.have_elems && ja.elems >= 128) || (ja.cull_on && ja.tris >= 48 && ja.r7 <= 30000))
            ja.claim_all = 1;
        j_groups_chunk(&ja, 0, (uint32_t)(ja.groups > 0 ? ja.groups : 0));
    }
    /* The pool's workers store to this context concurrently and their
     * ls_dirty updates may race. When the pages they can write are known
     * (the two-phase path), add exactly those; otherwise claim the whole
     * store, as this always used to -- which made runtime/spu/spurs_job.c
     * re-stage all 256 KB after every call (~5% of the process in a battle).
     * J187280_DIRTY_ALL=1 restores the old claim. */
    static int dirty_all = -1;
    if (dirty_all < 0) { const char *e = getenv("J187280_DIRTY_ALL"); dirty_all = e && atoi(e) == 1; }
    if (ja.claim_all || dirty_all) ctx->ls_dirty = ~0ull;
    else ctx->ls_dirty |= ja.dirty_extra;
    if (js_on) {
        const uint64_t now = js_now();
        atomic_fetch_add(&js_ns, now - js_start);
        js_report(now);
    }
}

