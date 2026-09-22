# Wan 2.2 image-to-video

A video variant of the worker: `MODEL_TYPE=wan2.2-i2v` bakes
[Wan 2.2 I2V A14B](https://huggingface.co/Wan-AI/Wan2.2-I2V-A14B) and the custom nodes a Wan
image-to-video workflow needs. It is not published on Docker Hub — build it yourself with the
[`Build Wan 2.2 I2V image`](../.github/workflows/build-wan.yml) workflow, which pushes to GHCR.

The client sends the whole graph in `input.workflow`, exactly like every other variant. Nothing
about the API changes.

## What is in the image

Weights, ~35 GiB, all from the Comfy-Org repackages (Apache 2.0, no HF token needed):

| File | Directory | Size |
| --- | --- | --- |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | 13.31 GiB |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | 13.31 GiB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 6.27 GiB |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | `loras/` | 1.14 GiB |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | `loras/` | 1.14 GiB |
| `wan_2.1_vae.safetensors` | `vae/` | 0.24 GiB |

Wan 2.2 is a mixture of experts: the high-noise expert handles early steps and global
composition, the low-noise expert the later ones. Both are needed, and a LoRA generally has to
be applied to each separately. The lightx2v distill LoRAs are what make 4-step sampling viable.

Custom nodes:

| Pack | Provides |
| --- | --- |
| `comfyui-kjnodes` | `ScheduledCFGGuidance`, `ModelPassThrough`, `VRAM_Debug`, `DummyOut`, `INTConstant`, `FloatConstant` |
| `comfyui-videohelpersuite` | `VHS_VideoCombine` — the MP4 writer |
| `comfyui-frame-interpolation` | `RIFE VFI` plus `rife49.pth`, which doubles 16 fps to 32 fps |

`EasyCache` is **not** in that list: it became a core ComfyUI node in 0.34. It is worth keeping
in a workflow — pure PyTorch, no special hardware, and it skips redundant steps.

### Deliberately left out

`SageAttention` and `TorchCompile` are both speed optimizations on a pipeline that is correct
without them, and both cost more than they look like they do:

- **SageAttention** compiles CUDA kernels at build time, and its
  `sageattn_qk_int8_pv_fp8_cuda++` path needs sm_89 or newer. That is what produces
  `fp8e4nv not supported in this architecture` on Ampere. Leaving it out keeps A40 / RTX A6000 /
  RTX 3090 usable and widens GPU availability considerably.
- **TorchCompile** costs 1-3 minutes of inductor compilation on every cold container, and
  recompiles per resolution when configured with `dynamic: false`.

Add them back deliberately, and measure.

## Endpoint settings

| Setting | Value | Why |
| --- | --- | --- |
| Container disk | **≥ 60 GB** | The default 20 GB cannot hold a ~45 GB image. |
| GPU | 24 GB is enough | 4090 / A40 / A6000 / L40S. Without SageAttention, Ampere works. |
| Network volume | optional | Only for swappable LoRAs, see below. |

The first job on a cold worker pays for loading both 14B experts, so allow several minutes
before deciding something is wrong.

## Adding LoRAs without rebuilding

The baked weights never change, but LoRAs are exactly the thing you end up swapping and
re-tuning. Rebuilding for each one means a 40-70 minute CI run and a ~45 GB push.

[`src/extra_model_paths.yaml`](../src/extra_model_paths.yaml) already maps `loras` to
`/runpod-volume/models/loras/`, and ComfyUI unions that with the baked `models/loras` — both are
visible at once. So attach a small network volume, drop `.safetensors` files into
`models/loras/` on it from a cheap Pod, and reference them by filename in the workflow. No
rebuild.

Two things to get right when stacking a LoRA on top of the 4-step distill:

- **Apply it to both experts, at different strengths.** A typical graph already runs the
  lightx2v LoRA at ~0.4 on the high-noise expert and 1.0 on the low-noise one. Chain a second
  `LoraLoaderModelOnly` per expert and start conservative — high ≈ 0.3-0.5, low ≈ 0.6-0.8. High-
  noise LoRAs drive global composition and in image-to-video readily override the start frame,
  and a second LoRA at full strength on top of a distill LoRA tends to cost motion.
- **Check the licence.** The base weights here are Apache 2.0 and the node packs are GPL-family
  or MIT, all fine commercially. Community LoRAs frequently ship with no licence at all, or with
  a Civitai licence that restricts commercial use.

## Verifying a workflow before you run it

A missing custom node shows up as a failed job on a paid GPU. Boot the image and diff your
workflow's classes against what ComfyUI actually registered:

```bash
curl -s localhost:8188/object_info | python3 -c '
import json, sys
have = set(json.load(sys.stdin))
want = {n["class_type"] for n in json.load(open("your_workflow.json")).values()}
print("MISSING:", want - have or "none")'
```

It must print `none`. Build with `--platform linux/amd64`: an ARM build silently fails on
RunPod.

## Video output

Nothing special is needed on the client. `VHS_VideoCombine` reports its MP4 under the `gifs`
key rather than `images`, and the handler collects every key in `MEDIA_KEYS`
(`images`, `gifs`, `videos`, `audio`) into the same output list. The clip comes back in
`output.images` as base64 with its real `.mp4` filename, or as an S3 URL if
`BUCKET_ENDPOINT_URL` is configured.
