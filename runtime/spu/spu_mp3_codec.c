/* Host MP3 codec for the MultiStream PIC seam (notes 039/040).
 *
 * Comet Crash's audio is all MP3, decoded on the real console by
 * Sony's MultiStream SPU task calling a codec plugin it loads from
 * /dev_flash/sys/external/flashMP3.pic into LS 0x1A900. This project
 * uses no Sony firmware, so the plugin's function is implemented
 * here on the host instead: when the SPU runtime's indirect-branch
 * dispatch reaches the (empty) codec entry, spu_channels.c calls
 * ps3_spu_mp3_pic_invoke(), which answers with the protocol the
 * lifted MultiStream code provably expects:
 *
 *   - in:  MP3 bytes staged by the caller at LS[r3]; param block at
 *          LS[r4]; per-stream workspace pointer in r7 (state key).
 *   - out: param+0x00 (BE u32) = input bytes consumed -- written on
 *          EVERY return; param+0x10 (BE u32) = PCM bytes in the
 *          returned buffer; r3 = LS address of the PCM (in the
 *          codec-owned range at 0x1C900), or 0 when no complete
 *          frame was available ("consumed but produced nothing").
 *   - Site A (lr=0x6E2C) primes: its returned pointer is ignored,
 *     so a frame decoded there is held and handed out again at the
 *     following site-B call (lr=0x06EB0), which is the one the
 *     caller DMA's from. r3 == -1 is never returned (it makes the
 *     caller dump diagnostics and abandon the stream for the pass).
 *
 * Decoding is minimp3 (public domain, single header, vendored
 * beside this file) -- the same decoder the project's native port
 * uses for these assets. No Sony code or firmware file is involved.
 * PS3_NO_HOST_CODEC=1 restores the old stub-return-0 behaviour.
 * PS3_CODEC_LOG=1 logs the first 16 invocations.
 */
#include "spu_context.h"

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define MINIMP3_IMPLEMENTATION
#include "minimp3.h"

#define CODEC_OUT_LS   0x1C900u   /* PIC BSS range, codec-owned */
#define CODEC_IN_MAX   8192u      /* staged-input window cap */
#define CODEC_MAX_STREAMS 8

typedef struct {
    uint32_t key;                 /* r7 stream workspace pointer */
    int      used;
    mp3dec_t dec;
    int16_t  hold[1152 * 2];      /* one max-size stereo frame */
    int      hold_bytes;
} codec_stream;

static codec_stream s_streams[CODEC_MAX_STREAMS];

static void ls_wr32(spu_context *ctx, uint32_t addr, uint32_t v)
{
    addr &= SPU_LS_MASK;
    ctx->ls[addr] = (uint8_t)(v >> 24);
    ctx->ls[addr + 1] = (uint8_t)(v >> 16);
    ctx->ls[addr + 2] = (uint8_t)(v >> 8);
    ctx->ls[addr + 3] = (uint8_t)v;
}

static codec_stream *stream_for(uint32_t key)
{
    int i;
    for (i = 0; i < CODEC_MAX_STREAMS; i++)
        if (s_streams[i].used && s_streams[i].key == key)
            return &s_streams[i];
    for (i = 0; i < CODEC_MAX_STREAMS; i++)
        if (!s_streams[i].used) {
            s_streams[i].used = 1;
            s_streams[i].key = key;
            s_streams[i].hold_bytes = 0;
            mp3dec_init(&s_streams[i].dec);
            return &s_streams[i];
        }
    return NULL;
}

static int codec_log(void)
{
    static int on = -1;
    if (on < 0) on = getenv("PS3_CODEC_LOG") ? 1 : 0;
    return on;
}

/* Returns 1 when the call was handled (caller performs SPU_RET with
 * the r3 left in place), 0 to fall back to the zeroed-PIC stub. */
int ps3_spu_mp3_pic_invoke(spu_context *ctx, uint32_t entry)
{
    static int disabled = -1;
    if (disabled < 0) disabled = getenv("PS3_NO_HOST_CODEC") ? 1 : 0;
    if (disabled) return 0;
    (void)entry;

    const uint32_t cursor = ctx->gpr[3]._u32[0] & SPU_LS_MASK;
    const uint32_t param  = ctx->gpr[4]._u32[0] & SPU_LS_MASK;
    const uint32_t key    = ctx->gpr[7]._u32[0] & SPU_LS_MASK;
    const uint32_t lr     = ctx->gpr[0]._u32[0] & SPU_LS_MASK;
    const int site_b = (lr == 0x6EB0u);

    codec_stream *st = stream_for(key);
    if (!st) return 0;

    uint32_t consumed = 0;

    if (!st->hold_bytes) {
        uint32_t avail = SPU_LS_SIZE - cursor;
        if (avail > CODEC_IN_MAX) avail = CODEC_IN_MAX;
        mp3dec_frame_info_t info;
        memset(&info, 0, sizeof(info));
        int samples = mp3dec_decode_frame(&st->dec, &ctx->ls[cursor],
                                          (int)avail, st->hold, &info);
        if (samples > 0 && info.frame_bytes > 0) {
            st->hold_bytes = samples * info.channels * (int)sizeof(int16_t);
            consumed = (uint32_t)info.frame_bytes;
            if (codec_log()) {
                static int n = 0;
                if (n < 16) { n++;
                    fprintf(stderr, "[host-codec] key=0x%X lr=0x%X frame: %d samples/ch, %d ch, %d Hz, consumed %d, pcm %d B\n",
                            key, lr, samples, info.channels, info.hz,
                            info.frame_bytes, st->hold_bytes);
                }
            }
        }
    }

    /* Consumed is an out-field on every return path. */
    ls_wr32(ctx, param + 0x00, consumed);

    if (st->hold_bytes) {
        memcpy(&ctx->ls[CODEC_OUT_LS], st->hold, (size_t)st->hold_bytes);
        ls_wr32(ctx, param + 0x10, (uint32_t)st->hold_bytes);
        ctx->gpr[3]._u32[0] = CODEC_OUT_LS;
        if (site_b) st->hold_bytes = 0;   /* site A peeks; site B takes */
    } else {
        ctx->gpr[3]._u32[0] = 0;
    }
    return 1;
}
