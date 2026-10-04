/**
 * conv2d_bn.h — 2-D convolution with folded BatchNorm, float32
 *
 * After BN-folding (see stm32/bn_fold.py) each conv block has only:
 *   weight[Cout, Cin, kH, kW]  and  bias[Cout]
 * No separate BN pass is needed at inference.
 *
 * Layout conventions (row-major, matching PyTorch default):
 *   input  / output : [C, H, W]   — channel-first
 *   weight          : [Cout, Cin, kH, kW]
 *
 * Limitations (sufficient for SpikeResNet10 on CIFAR-10):
 *   - Square kernels only (kH == kW)
 *   - Symmetric padding
 *   - Dilation == 1
 *   - Groups == 1
 */

#pragma once
#include <stdint.h>

/**
 * conv2d
 *
 * @param in      Input  feature map [Cin × H_in  × W_in]
 * @param out     Output feature map [Cout × H_out × W_out]  (caller-allocated)
 * @param weight  Kernel             [Cout × Cin  × k  × k]
 * @param bias    Bias               [Cout]  (may be NULL for bias=False layers)
 * @param Cin     Input channels
 * @param Cout    Output channels
 * @param H_in    Input height
 * @param W_in    Input width
 * @param k       Kernel size (square)
 * @param stride  Convolution stride
 * @param pad     Zero-padding size
 *
 * H_out = (H_in + 2*pad - k) / stride + 1
 * W_out = (W_in + 2*pad - k) / stride + 1
 */
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
            uint32_t     pad);

/**
 * maxpool2d — 2×2 max-pooling, stride 2 (most common in SpikeResNet).
 *
 * @param in   Input  [C × H_in  × W_in]
 * @param out  Output [C × H_out × W_out]
 * H_out = H_in / 2,  W_out = W_in / 2
 */
void maxpool2d_2x2(const float *in,
                   float       *out,
                   uint32_t     C,
                   uint32_t     H_in,
                   uint32_t     W_in);

/**
 * adaptive_maxpool2d_1x1 — global max-pooling per channel.
 * Reduces [C × H × W] → [C × 1 × 1].
 */
void adaptive_maxpool2d_1x1(const float *in,
                             float       *out,
                             uint32_t     C,
                             uint32_t     H,
                             uint32_t     W);

/**
 * linear — fully-connected layer.
 *
 * @param in     Input  [Cin]
 * @param out    Output [Cout]
 * @param weight [Cout × Cin]
 * @param bias   [Cout]
 */
void linear(const float *in,
            float       *out,
            const float *weight,
            const float *bias,
            uint32_t     Cin,
            uint32_t     Cout);
