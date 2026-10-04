# SpikeResNet10 → STM32F769 Deployment

## Target hardware

| Resource | STM32F769I-DISCO |
|---|---|
| CPU | ARM Cortex-M7 @ 216 MHz, FPU (float32) |
| Internal Flash | 2 MB |
| Internal SRAM | 512 KB (DTCM 128 KB + SRAM1/2/3 384 KB) |
| External SDRAM | 16 MB @ 0xC0000000 |
| QSPI NOR Flash | up to 128 MB (board dependent) |

## Memory budget

| Item | float32 | int8 |
|---|---|---|
| SpikeResNet10 weights (BN-folded) | ~18.6 MB | ~4.9 MB |
| Peak activations (batch=1) | ~512 KB | ~128 KB |
| Membrane state buffers | ~1 KB | — |

**Conclusion:** weights must live in external SDRAM (16 MB).  
Activation buffers fit in internal SRAM at float32; DTCM (128 KB) covers all layers at int8.

---

## Repository layout

```
stm32/
├── model_info.py          — parameter count & per-layer breakdown
├── bn_fold.py             — fold Conv+BN into a single Conv with bias
├── export_weights.py      — export trained .pth → weights.bin + weights.h/.c
├── validate_export.py     — numerical check: bin output matches PyTorch
└── c_src/
    ├── include/
    │   ├── lif.h              — LIF neuron API
    │   ├── conv2d_bn.h        — Conv2D / pooling / linear API
    │   └── spikeresnet10.h    — network context + forward declaration
    └── src/
        ├── lif.c              — mem = β·mem + cur, fire, zero-reset
        ├── conv2d_bn.c        — pure-C float32 kernels
        └── spikeresnet10.c    — full layer-by-layer forward pass
```

---

## Step-by-step procedure

### Step 1 — Check memory fit

```bash
python stm32/model_info.py
```

Prints parameter count, weight size in float32 / int8, and fit assessment against STM32F769 memory regions.

---

### Step 2 — Export weights from a trained checkpoint

```bash
python stm32/export_weights.py \
    --weights  weights/spike/exp1/resnet10_weights_CIFAR10_T_1_E_100_L_mse_count_loss_A_True_S_42.pth \
    --dataset  CIFAR10 \
    --out      stm32/c_src/weights
```

Produces:
- `stm32/c_src/weights/weights.bin` — flat float32 blob (layer order matches `spikeresnet10.c`)
- `stm32/c_src/weights/weights.h`   — `extern const float …` declarations + byte offsets
- `stm32/c_src/weights/weights.c`   — array definitions tagged `__attribute__((section(".sdram_weights")))`

The BN fold is applied automatically: every Conv+BN pair is collapsed to a single Conv with bias, removing all BN computation from the MCU.

---

### Step 3 — Validate the export

```bash
python stm32/validate_export.py \
    --weights      weights/spike/exp1/resnet10_weights_CIFAR10_T_1_E_100_L_mse_count_loss_A_True_S_42.pth \
    --weights-bin  stm32/c_src/weights/weights.bin \
    --dataset      CIFAR10
```

Runs the PyTorch model and the binary-loaded model on the same 64 CIFAR-10 test images and reports max absolute error and top-1 prediction agreement. Expected output:

```
Max absolute error : < 1e-5
Top-1 prediction   : MATCH
EXPORT VALIDATED
```

---

### Step 4 — STM32CubeIDE project setup

1. Create a new STM32CubeIDE project for **STM32F769NIHx**.
2. Enable external SDRAM in CubeMX (FMC, IS42S32400F, mapped at `0xC0000000`).
3. Add to the project:
   - `stm32/c_src/src/*.c`
   - `stm32/c_src/include/` (as include path)
   - `stm32/c_src/weights/weights.c`
4. Edit the linker script to place `.sdram_weights` in SDRAM:
   ```
   .sdram_weights (NOLOAD) :
   {
       *(.sdram_weights)
   } >SDRAM
   ```
   Remove `NOLOAD` if you want the weights burned into QSPI flash and copied to SDRAM at startup by the startup code.

5. In your application `main.c`:
   ```c
   #include "spikeresnet10.h"

   static SpikeResNet10Ctx net_ctx;
   static float logits[SPIKERESNET10_NUM_CLASSES];

   // Initialise once after SDRAM is ready
   spikeresnet10_init(&net_ctx);

   // Per-image inference (input: float32 CHW, normalised)
   spikeresnet10_forward(&net_ctx, input_image_chw, logits);
   int label = argmax_f32(logits, SPIKERESNET10_NUM_CLASSES);
   ```

---

### Step 5 — Input preprocessing on the MCU

CIFAR-10 normalisation (must match training):

```c
// mean = {0.4914, 0.4822, 0.4465}
// std  = {0.2023, 0.1994, 0.2010}
static const float mean[3] = {0.4914f, 0.4822f, 0.4465f};
static const float std[3]  = {0.2023f, 0.1994f, 0.2010f};

for (int c = 0; c < 3; c++)
    for (int i = 0; i < 32*32; i++)
        input_f32[c*32*32 + i] =
            (raw_uint8[c*32*32 + i] / 255.0f - mean[c]) / std[c];
```

---

## Roadmap / next steps

| Priority | Task | Benefit |
|---|---|---|
| 1 | Replace `conv2d_bn.c` with CMSIS-NN `arm_convolve_f32` | 2–4× speedup via Cortex-M7 SIMD |
| 2 | Post-training int8 quantisation (per-channel) | Weights 4.9 MB → fits in SDRAM with headroom |
| 3 | Replace `conv2d_bn.c` with CMSIS-NN `arm_convolve_s8` | Further 2–4× speedup; int8 activations |
| 4 | Measure inference latency and power on hardware | Needed for publication / deployment spec |
| 5 | Multi-timestep support (T > 1) | Higher accuracy; reuse same `SpikeResNet10Ctx` across calls |
