/**
 * spikeresnet10.c — SpikeResNet10 forward pass (float32, T=1)
 *
 * Mirrors the Python forward() in models/spikeresnet.py SpikeResNet10Model.
 *
 * Activation buffers A and B are allocated statically to avoid stack overflow.
 * They are reused (ping-pong) across layers.  Buffer size must cover the
 * largest feature map in the network: 64 × 32 × 32 = 65 536 floats (256 KB).
 */

#include "spikeresnet10.h"
#include "conv2d_bn.h"
#include "lif.h"
#include <string.h>

/* ── Static ping-pong activation buffers ────────────────────────────────── */
/* Placed in .dtcm_data section for fastest access on Cortex-M7.            */
/* Adjust linker section name to match your STM32CubeIDE linker script.     */

#define DTCM_SECTION __attribute__((section(".dtcm_data")))

DTCM_SECTION static float act_A[ACT_BUF_SIZE];
DTCM_SECTION static float act_B[ACT_BUF_SIZE];

/* Spike and skip-connection buffers share the same sizes */
DTCM_SECTION static float spk_A[ACT_BUF_SIZE];
DTCM_SECTION static float spk_B[ACT_BUF_SIZE];
DTCM_SECTION static float skip [ACT_BUF_SIZE];   /* identity / downsample output */

/* ── Helpers ─────────────────────────────────────────────────────────────── */

static inline uint32_t chw(uint32_t c, uint32_t h, uint32_t w)
{
    return c * h * w;
}

/* ── Init ────────────────────────────────────────────────────────────────── */

void spikeresnet10_init(SpikeResNet10Ctx *ctx)
{
    memset(ctx, 0, sizeof(SpikeResNet10Ctx));
}

/* ── Forward pass ────────────────────────────────────────────────────────── */

void spikeresnet10_forward(SpikeResNet10Ctx *ctx,
                            const float      *input,
                            float            *logits)
{
    const float beta = SPIKERESNET10_BETA;
    const float thr  = SPIKERESNET10_THRESHOLD;

    /* ── block1: Conv(3→64, 3×3, s=1, p=1) + LIF ─────────────────────────
     *   input   : [3,  32, 32]
     *   act_A   : [64, 32, 32]   conv output (current)
     *   spk_A   : [64, 32, 32]   spikes
     */
    conv2d(input, act_A,
           block1_weight, block1_bias,
           3, 64, 32, 32, 3, 1, 1);
    lif_step(act_A, ctx->mem1, spk_A, chw(64,32,32), beta, thr);

    /* ── resBlock2_1: Conv(64→64, 3×3) + LIF ─────────────────────────────
     *   spk_A → act_B → spk_B
     */
    conv2d(spk_A, act_B,
           resBlock2_1_weight, resBlock2_1_bias,
           64, 64, 32, 32, 3, 1, 1);
    lif_step(act_B, ctx->mem2_1, spk_B, chw(64,32,32), beta, thr);

    /* ── resBlock2_2: Conv(64→64, 3×3) + LIF → skip-add with spk_A ───────
     *   spk_B → act_A → spk_A (residual output, clamped + spk_A from block1)
     */
    conv2d(spk_B, act_A,
           resBlock2_2_weight, resBlock2_2_bias,
           64, 64, 32, 32, 3, 1, 1);
    lif_step(act_A, ctx->mem2_2, spk_B, chw(64,32,32), beta, thr);
    /* skip-add: spk_r2 = clamp(spk_B + spk_A, 1)  — spk_A is block1 output */
    lif_spike_add_clamp(spk_B, spk_A, spk_A, chw(64,32,32));
    /* spk_A now holds spk_r2  [64, 32, 32] */

    /* ── downsample3: Conv(64→128, 1×1, s=2) + LIF (identity branch) ─────
     *   spk_A → act_B → skip  [128, 16, 16]
     */
    conv2d(spk_A, act_B,
           downsample3_weight, downsample3_bias,
           64, 128, 32, 32, 1, 2, 0);
    lif_step(act_B, ctx->mem_d3, skip, chw(128,16,16), beta, thr);

    /* ── resBlock4_1: Conv(64→128, 3×3, s=2) + LIF ───────────────────────
     *   spk_A [64,32,32] → act_B [128,16,16] → spk_B
     */
    conv2d(spk_A, act_B,
           resBlock4_1_weight, resBlock4_1_bias,
           64, 128, 32, 32, 3, 2, 1);
    lif_step(act_B, ctx->mem4_1, spk_B, chw(128,16,16), beta, thr);

    /* ── resBlock4_2: Conv(128→128, 3×3) + LIF → skip-add with skip ──────
     *   spk_B [128,16,16] → act_B → spk_A [128,16,16]
     */
    conv2d(spk_B, act_B,
           resBlock4_2_weight, resBlock4_2_bias,
           128, 128, 16, 16, 3, 1, 1);
    lif_step(act_B, ctx->mem4_2, spk_A, chw(128,16,16), beta, thr);
    lif_spike_add_clamp(spk_A, skip, spk_A, chw(128,16,16));
    /* spk_A = spk_r4  [128, 16, 16] */

    /* ── downsample5: Conv(128→256, 1×1, s=2) + LIF ──────────────────────
     *   spk_A [128,16,16] → act_B [256,8,8] → skip
     */
    conv2d(spk_A, act_B,
           downsample5_weight, downsample5_bias,
           128, 256, 16, 16, 1, 2, 0);
    lif_step(act_B, ctx->mem_d5, skip, chw(256,8,8), beta, thr);

    /* ── resBlock6_1: Conv(128→256, 3×3, s=2) + LIF ──────────────────────
     *   spk_A [128,16,16] → act_B [256,8,8] → spk_B
     */
    conv2d(spk_A, act_B,
           resBlock6_1_weight, resBlock6_1_bias,
           128, 256, 16, 16, 3, 2, 1);
    lif_step(act_B, ctx->mem6_1, spk_B, chw(256,8,8), beta, thr);

    /* ── resBlock6_2: Conv(256→256, 3×3) + LIF → skip-add ────────────────
     *   spk_B → act_B → spk_A [256,8,8]
     */
    conv2d(spk_B, act_B,
           resBlock6_2_weight, resBlock6_2_bias,
           256, 256, 8, 8, 3, 1, 1);
    lif_step(act_B, ctx->mem6_2, spk_A, chw(256,8,8), beta, thr);
    lif_spike_add_clamp(spk_A, skip, spk_A, chw(256,8,8));
    /* spk_A = spk_r6  [256, 8, 8] */

    /* ── downsample7: Conv(256→512, 1×1, s=2) + LIF ──────────────────────
     *   spk_A [256,8,8] → act_B [512,4,4] → skip
     */
    conv2d(spk_A, act_B,
           downsample7_weight, downsample7_bias,
           256, 512, 8, 8, 1, 2, 0);
    lif_step(act_B, ctx->mem_d7, skip, chw(512,4,4), beta, thr);

    /* ── resBlock8_1: Conv(256→512, 3×3, s=2) + LIF ──────────────────────
     *   spk_A [256,8,8] → act_B [512,4,4] → spk_B
     */
    conv2d(spk_A, act_B,
           resBlock8_1_weight, resBlock8_1_bias,
           256, 512, 8, 8, 3, 2, 1);
    lif_step(act_B, ctx->mem8_1, spk_B, chw(512,4,4), beta, thr);

    /* ── resBlock8_2: Conv(512→512, 3×3) + LIF → skip-add ────────────────
     *   spk_B → act_B → spk_A [512,4,4]
     */
    conv2d(spk_B, act_B,
           resBlock8_2_weight, resBlock8_2_bias,
           512, 512, 4, 4, 3, 1, 1);
    lif_step(act_B, ctx->mem8_2, spk_A, chw(512,4,4), beta, thr);
    lif_spike_add_clamp(spk_A, skip, spk_A, chw(512,4,4));
    /* spk_A = spk_r8  [512, 4, 4] */

    /* ── amax9: AdaptiveMaxPool2d(1) ──────────────────────────────────────
     *   spk_A [512,4,4] → spk_B [512]
     */
    adaptive_maxpool2d_1x1(spk_A, spk_B, 512, 4, 4);

    /* ── fc9: Linear(512→NUM_CLASSES) ────────────────────────────────────
     *   spk_B [512] → act_A [NUM_CLASSES]
     */
    linear(spk_B, act_A,
           fc9_weight, fc9_bias,
           512, SPIKERESNET10_NUM_CLASSES);

    /* ── lifOut: LIF (no zero-reset, output=True → return membrane) ───────
     *   PyTorch uses reset_mechanism='zero' + output=True for this layer,
     *   meaning the membrane voltage (before reset) is returned as the logit.
     *   We compute mem = beta*mem + cur, fire, zero-reset, then return mem
     *   BEFORE reset to match PyTorch output=True behaviour.
     */
    for (uint32_t i = 0; i < (uint32_t)SPIKERESNET10_NUM_CLASSES; i++) {
        float m = beta * ctx->mem_out[i] + act_A[i];
        float s = (m > thr) ? 1.0f : 0.0f;
        logits[i]      = m;               /* return pre-reset membrane */
        ctx->mem_out[i] = m * (1.0f - s); /* zero-reset for next step  */
    }
}

/* ── Argmax ──────────────────────────────────────────────────────────────── */

int argmax_f32(const float *v, uint32_t n)
{
    int   best_i = 0;
    float best_v = v[0];
    for (uint32_t i = 1; i < n; i++) {
        if (v[i] > best_v) { best_v = v[i]; best_i = (int)i; }
    }
    return best_i;
}
