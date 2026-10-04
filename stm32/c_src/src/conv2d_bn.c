/**
 * conv2d_bn.c — float32 convolution, pooling and linear layers
 *
 * Pure C implementation targeting ARM Cortex-M7 with FPU.
 * For production, replace conv2d() with CMSIS-NN arm_convolve_f32()
 * or quantise to int8 and use arm_convolve_s8().
 */

#include "conv2d_bn.h"
#include <float.h>   /* FLT_MAX */
#include <string.h>  /* memset  */

/* ── Convolution ─────────────────────────────────────────────────────────── */

void conv2d(const float *in,
            float       *out,
            const float *weight,
            const float *bias,
            uint32_t     Cin,
            uint32_t     Cout,
            uint32_t     H_in,
            uint32_t     W_in,
            uint32_t     k,
            uint32_t     stride,
            uint32_t     pad)
{
    const uint32_t H_out = (H_in + 2 * pad - k) / stride + 1;
    const uint32_t W_out = (W_in + 2 * pad - k) / stride + 1;

    for (uint32_t oc = 0; oc < Cout; oc++) {
        float b = (bias != NULL) ? bias[oc] : 0.0f;
        for (uint32_t oh = 0; oh < H_out; oh++) {
            for (uint32_t ow = 0; ow < W_out; ow++) {
                float acc = b;
                for (uint32_t ic = 0; ic < Cin; ic++) {
                    for (uint32_t kh = 0; kh < k; kh++) {
                        for (uint32_t kw = 0; kw < k; kw++) {
                            int32_t ih = (int32_t)(oh * stride + kh) - (int32_t)pad;
                            int32_t iw = (int32_t)(ow * stride + kw) - (int32_t)pad;
                            if (ih >= 0 && ih < (int32_t)H_in &&
                                iw >= 0 && iw < (int32_t)W_in) {
                                /* in  [ic, H_in, W_in] */
                                float x = in[ic * H_in * W_in +
                                             (uint32_t)ih * W_in +
                                             (uint32_t)iw];
                                /* weight [oc, ic, k, k] */
                                float w = weight[oc * Cin * k * k +
                                                 ic * k * k +
                                                 kh * k +
                                                 kw];
                                acc += x * w;
                            }
                        }
                    }
                }
                out[oc * H_out * W_out + oh * W_out + ow] = acc;
            }
        }
    }
}

/* ── MaxPool 2×2, stride 2 ───────────────────────────────────────────────── */

void maxpool2d_2x2(const float *in,
                    float       *out,
                    uint32_t     C,
                    uint32_t     H_in,
                    uint32_t     W_in)
{
    const uint32_t H_out = H_in / 2;
    const uint32_t W_out = W_in / 2;

    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < H_out; oh++) {
            for (uint32_t ow = 0; ow < W_out; ow++) {
                float m = -FLT_MAX;
                for (uint32_t ph = 0; ph < 2; ph++) {
                    for (uint32_t pw = 0; pw < 2; pw++) {
                        float v = in[c * H_in * W_in +
                                     (oh * 2 + ph) * W_in +
                                     (ow * 2 + pw)];
                        if (v > m) m = v;
                    }
                }
                out[c * H_out * W_out + oh * W_out + ow] = m;
            }
        }
    }
}

/* ── Adaptive max-pool → 1×1 ─────────────────────────────────────────────── */

void adaptive_maxpool2d_1x1(const float *in,
                              float       *out,
                              uint32_t     C,
                              uint32_t     H,
                              uint32_t     W)
{
    const uint32_t spatial = H * W;
    for (uint32_t c = 0; c < C; c++) {
        float m = -FLT_MAX;
        const float *plane = in + c * spatial;
        for (uint32_t i = 0; i < spatial; i++) {
            if (plane[i] > m) m = plane[i];
        }
        out[c] = m;
    }
}

/* ── Linear ──────────────────────────────────────────────────────────────── */

void linear(const float *in,
            float       *out,
            const float *weight,
            const float *bias,
            uint32_t     Cin,
            uint32_t     Cout)
{
    for (uint32_t oc = 0; oc < Cout; oc++) {
        float acc = (bias != NULL) ? bias[oc] : 0.0f;
        const float *row = weight + oc * Cin;
        for (uint32_t ic = 0; ic < Cin; ic++) {
            acc += row[ic] * in[ic];
        }
        out[oc] = acc;
    }
}
