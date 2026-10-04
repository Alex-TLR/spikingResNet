/**
 * lif.c — Leaky Integrate-and-Fire neuron implementation (float32, zero-reset)
 */

#include "lif.h"
#include <string.h>
#include <math.h>

void lif_step(const float *cur,
              float       *mem,
              float       *spk,
              uint32_t     n,
              float        beta,
              float        threshold)
{
    for (uint32_t i = 0; i < n; i++) {
        float m = beta * mem[i] + cur[i];
        float s = (m > threshold) ? 1.0f : 0.0f;
        mem[i]  = m * (1.0f - s);   /* zero-reset */
        spk[i]  = s;
    }
}

void lif_spike_add_clamp(const float *a,
                          const float *b,
                          float       *out,
                          uint32_t     n)
{
    for (uint32_t i = 0; i < n; i++) {
        float v = a[i] + b[i];
        out[i]  = (v > 1.0f) ? 1.0f : v;
    }
}

void lif_mem_init(float *mem, uint32_t n)
{
    memset(mem, 0, n * sizeof(float));
}
