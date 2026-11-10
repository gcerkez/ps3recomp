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
    int      fmt_hz;              /* CODEC FIX2: last decoded frame format */
    int      fmt_ch;
    uint8_t  tail[4096];          /* CODEC FIX2: recent consumed frames */
    int      tail_len;
    int      tail_fl[24];         /* whole-frame lengths queued in tail */
    int      tail_nf;
} codec_stream;

static codec_stream s_streams[CODEC_MAX_STREAMS];

static void ls_wr32(spu_context *ctx, uint32_t addr, uint32_t v)
{
    addr &= SPU_LS_MASK;
    spu_ls_mark(ctx, addr, 4);
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
            s_streams[i].fmt_hz = 0;
            s_streams[i].fmt_ch = 0;
            s_streams[i].tail_len = 0;
            s_streams[i].tail_nf = 0;
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
/* CODEC FIX2: frame length + format from the 4-byte MP3 header at p,
 * or 0 when the header is not a parseable Layer III header. The seam
 * accepts a decode only when minimp3's reported byte count matches
 * this - a decode that overruns into stale staged bytes reports a
 * longer "frame" than the header allows and must not be consumed. */
static int mp3_frame_len_at(const uint8_t *p, int avail, int *hz_out,
                            int *ch_out, int *spf_out)
{
    static const int br1[16] = {0,32,40,48,56,64,80,96,112,128,160,192,224,256,320,-1};
    static const int br2[16] = {0,8,16,24,32,40,48,56,64,80,96,112,128,144,160,-1};
    static const int sr_tab[3][3] = {{44100,48000,32000},{22050,24000,16000},{11025,12000,8000}};
    int ver, bri, br, sri, sr;
    if (avail < 4) return 0;
    if (p[0] != 0xFF || (p[1] & 0xE0) != 0xE0) return 0;
    ver = (p[1] >> 3) & 3;              /* 3=MPEG1, 2=MPEG2, 0=MPEG2.5 */
    if (ver == 1) return 0;
    if (((p[1] >> 1) & 3) != 1) return 0;   /* Layer III only */
    bri = (p[2] >> 4) & 15;
    br = (ver == 3) ? br1[bri] : br2[bri];
    sri = (p[2] >> 2) & 3;
    if (br <= 0 || sri == 3) return 0;
    sr = sr_tab[ver == 3 ? 0 : (ver == 2 ? 1 : 2)][sri];
    if (hz_out) *hz_out = sr;
    if (ch_out) *ch_out = ((p[3] >> 6) == 3) ? 1 : 2;
    if (spf_out) *spf_out = (ver == 3) ? 1152 : 576;
    return ((ver == 3) ? 144 : 72) * br * 1000 / sr + ((p[2] >> 1) & 1);
}

/* CODEC FIX2: remember a consumed frame so a later recovery can
 * rebuild the bit reservoir by replaying whole frames in order. */
static void codec_tail_push(codec_stream *st, const uint8_t *bytes, int len)
{
    if (len <= 0 || len > (int)sizeof(st->tail)) { st->tail_len = 0; st->tail_nf = 0; return; }
    while ((st->tail_len + len > (int)sizeof(st->tail) || st->tail_nf >= 24) && st->tail_nf > 0) {
        int drop = st->tail_fl[0];
        memmove(st->tail, st->tail + drop, (size_t)(st->tail_len - drop));
        memmove(st->tail_fl, st->tail_fl + 1, (size_t)(st->tail_nf - 1) * sizeof(int));
        st->tail_len -= drop;
        st->tail_nf--;
    }
    memcpy(st->tail + st->tail_len, bytes, (size_t)len);
    st->tail_fl[st->tail_nf++] = len;
    st->tail_len += len;
}

/* SAVESTATE RESET: called by the title glue after a savestate
 * restore. The stream table is keyed by guest workspace pointer
 * and a same-format restart is never detected (notes 040); the
 * restored workspaces belong to the saved timeline, so drop
 * every stream -- stream_for() re-initialises slots on use. */
void ps3_spu_mp3_reset_streams(void)
{
    memset(s_streams, 0, sizeof s_streams);
    if (codec_log())
        fprintf(stderr, "[codec] savestate: stream table reset\n");
}

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
        /* CODEC FIX2: know the staged frame's true length up front. */
        int fh_hz = 0, fh_ch = 0, fh_spf = 0;
        int flen = mp3_frame_len_at(&ctx->ls[cursor], (int)avail,
                                    &fh_hz, &fh_ch, &fh_spf);
        /* A format change at this seam is a new stream: do not carry
         * the previous stream's bit reservoir into it. */
        if (flen > 0 && st->fmt_hz && (fh_hz != st->fmt_hz || fh_ch != st->fmt_ch)) {
            mp3dec_init(&st->dec);
            st->tail_len = 0;
            st->tail_nf = 0;
        }
        int samples = mp3dec_decode_frame(&st->dec, &ctx->ls[cursor],
                                          (int)avail, st->hold, &info);
        int ok = (samples > 0 && info.frame_bytes > 0 &&
                  (flen == 0 || info.frame_bytes == flen));
        if (!ok && flen > 0 && flen <= (int)avail) {
            /* Recovery: replay the kept tail frames through a fresh
             * decoder to rebuild the bit reservoir exactly, then
             * decode the staged frame. */
            uint8_t scratch[4096 + 2048];
            int slen = st->tail_len + flen;
            if (slen <= (int)sizeof(scratch)) {
                mp3dec_t fresh;
                mp3dec_frame_info_t fi2;
                int16_t tmp[1152 * 2];
                int pos = 0, guard = 0;
                memcpy(scratch, st->tail, (size_t)st->tail_len);
                memcpy(scratch + st->tail_len, &ctx->ls[cursor], (size_t)flen);
                mp3dec_init(&fresh);
                while (pos < st->tail_len && guard++ < 32) {
                    memset(&fi2, 0, sizeof(fi2));
                    if (mp3dec_decode_frame(&fresh, scratch + pos,
                                            slen - pos, tmp, &fi2) < 0 ||
                        fi2.frame_bytes <= 0)
                        break;
                    pos += fi2.frame_bytes;
                }
                if (pos == st->tail_len) {
                    memset(&fi2, 0, sizeof(fi2));
                    int s2 = mp3dec_decode_frame(&fresh, scratch + pos,
                                                 slen - pos, tmp, &fi2);
                    if (s2 > 0 && fi2.frame_bytes == flen) {
                        st->dec = fresh;
                        memcpy(st->hold, tmp,
                               (size_t)s2 * fi2.channels * sizeof(int16_t));
                        samples = s2;
                        info = fi2;
                        ok = 1;
                    }
                }
            }
            if (!ok) {
                /* Last resort: skip the frame. Consuming its header
                 * length and emitting silence keeps the caller's
                 * staging advancing - a stream must never wedge -
                 * and the skipped frame joins the tail so the next
                 * frame can heal through the recovery path. */
                memset(st->hold, 0, sizeof(st->hold));
                samples = fh_spf;
                memset(&info, 0, sizeof(info));
                info.frame_bytes = flen;
                info.hz = fh_hz;
                info.channels = fh_ch;
                mp3dec_init(&st->dec);
                ok = 1;
            }
        }
        if (ok) {
            if (flen > 0 && info.frame_bytes == flen)
                codec_tail_push(st, &ctx->ls[cursor], flen);
            st->fmt_hz = info.hz;
            st->fmt_ch = info.channels;
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
        /* The real PIC is SPU code: its PCM in LS is big-endian.
         * minimp3 hands us host-order int16, so byte-swap on the
         * way into LS (v1 shipped them swapped the wrong way --
         * the mixer pinned at full scale on byte-swapped samples). */
        { uint8_t *dst = &ctx->ls[CODEC_OUT_LS];
          int i, n = st->hold_bytes / 2;
          for (i = 0; i < n; i++) {
              uint16_t v = (uint16_t)st->hold[i];
              dst[2 * i] = (uint8_t)(v >> 8);
              dst[2 * i + 1] = (uint8_t)v;
          } }
        ls_wr32(ctx, param + 0x10, (uint32_t)st->hold_bytes);
        ctx->gpr[3]._u32[0] = CODEC_OUT_LS;
        if (site_b) st->hold_bytes = 0;   /* site A peeks; site B takes */
    } else {
        ctx->gpr[3]._u32[0] = 0;
    }
    return 1;
}
