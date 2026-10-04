/**
 * lif.h — Leaky Integrate-and-Fire neuron (float32, zero-reset)
 *
 * Matches snntorch.Leaky with reset_mechanism='zero':
 *
 *   mem  = beta * mem + cur_input
 *   spk  = (mem > threshold) ? 1.0f : 0.0f
 *   mem *= (1.0f - spk)          // zero-reset
 *
 * Usage
 * -----
 *   float mem[N] = {0};          // membrane state buffer (zero-initialised)
 *   float spk[N];
 *   lif_step(cur, mem, spk, N, BETA, THRESHOLD);
 *
 * All buffers must be allocated by the caller. No dynamic allocation.
 */

#pragma once
#include <stdint.h>

/* Apply one LIF timestep to a flat array of N neurons.
 *
 *  cur       [in]     synaptic input current,  length N
 *  mem       [in/out] membrane voltage,         length N  (updated in-place)
 *  spk       [out]    output spikes (0 or 1),   length N
 *  n         number of neurons
 *  beta      leak factor  (e.g. 0.5f)
 *  threshold firing threshold (e.g. 1.0f)
 */
void lif_step(const float *cur,
              float       *mem,
              float       *spk,
              uint32_t     n,
              float        beta,
              float        threshold);

/* Clamp-spike addition used by skip connections:
 *   out[i] = fminf(a[i] + b[i], 1.0f)
 */
void lif_spike_add_clamp(const float *a,
                         const float *b,
                         float       *out,
                         uint32_t     n);

/* Zero-initialise a membrane state buffer. */
void lif_mem_init(float *mem, uint32_t n);
