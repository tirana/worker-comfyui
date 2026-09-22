# worker-comfyui — Wan 2.2 image-to-video

A RunPod serverless worker that runs Wan 2.2 image-to-video ComfyUI workflows. Fork of
[runpod-workers/worker-comfyui](https://github.com/runpod-workers/worker-comfyui), stripped to
this one job.



Two differences from upstream:

- **Video output is returned.** Upstream only collected a node's `images`; `VHS_VideoCombine`
  reports its MP4 under `gifs`, so the clip was silently discarded. The handler now collects
  every key in `MEDIA_KEYS` — `images`, `gifs`, `videos`, `audio` — into one list.
- **Wan 2.2 weights and video nodes are baked in.** One image, no `MODEL_TYPE` variants.

## Build

The [`Build and Push Image`](.github/workflows/build-image.yml) workflow builds and pushes to
GHCR. Manual trigger only — ~45 GB and 40-70 minutes.

The resulting package is **private until you change it by hand** in the package settings, and
RunPod cannot pull it before then. Public also keeps it free of your storage quota.

## Endpoint settings

| Setting | Value | Why |
| --- | --- | --- |
| Container image | `ghcr.io/<owner>/worker-comfyui:<tag>` | |
| Container disk | **≥ 60 GB** | The default 20 GB cannot hold a 45 GB image. |
| GPU | 24 GB is enough | 4090 / A40 / A6000 / L40S. |
| Network volume | optional | Only for swappable LoRAs, see below. |

## API

Send the whole ComfyUI workflow in API format:

```json
{
  "input": {
    "workflow": { "...": "the graph" },
    "images": [{ "name": "input.png", "image": "<base64>" }]
  }
}
```

The clip comes back in `output.images` as base64 with its real `.mp4` filename, or as an S3
URL if `BUCKET_ENDPOINT_URL` is set.

Useful environment variables: `BUCKET_ENDPOINT_URL` / `BUCKET_ACCESS_KEY_ID` /
`BUCKET_SECRET_ACCESS_KEY` for S3 output, `REFRESH_WORKER=true` to restart after each job,
`COMFY_LOG_LEVEL`, and `NETWORK_VOLUME_DEBUG=true` to print what ComfyUI discovered on the
volume.

## What is in the image

Weights, ~35 GiB, from the Comfy-Org repackages (Apache 2.0, no HF token needed):

| File | Directory | Size |
| --- | --- | --- |
| `wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | 13.31 GiB |
| `wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors` | `diffusion_models/` | 13.31 GiB |
| `umt5_xxl_fp8_e4m3fn_scaled.safetensors` | `text_encoders/` | 6.27 GiB |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors` | `loras/` | 1.14 GiB |
| `wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors` | `loras/` | 1.14 GiB |
| `wan_2.1_vae.safetensors` | `vae/` | 0.24 GiB |

Wan 2.2 is a mixture of experts: the high-noise expert handles early steps and global
composition, the low-noise one the later steps. Both are needed, and a LoRA generally has to be
applied to each separately. The lightx2v distill LoRAs are what make 4-step sampling viable.

Custom nodes — a network volume cannot supply these, so they have to be in the image:

| Pack | Provides |
| --- | --- |
| [KJNodes](https://github.com/kijai/ComfyUI-KJNodes) | `ScheduledCFGGuidance`, `ModelPassThrough`, `VRAM_Debug`, `DummyOut`, `INTConstant`, `FloatConstant` |
| [VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite) | `VHS_VideoCombine` — the MP4 writer |
| [Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) | `RIFE VFI` plus `rife49.pth`, doubling 16 fps to 32 fps |

`EasyCache` is **not** in that list — it became a core ComfyUI node in 0.34. Worth keeping in a
workflow: pure PyTorch, no special hardware, skips redundant steps.

**Deliberately left out.** `SageAttention` compiles CUDA kernels at build time and its
`sageattn_qk_int8_pv_fp8_cuda++` path needs sm_89+, which is what produces
`fp8e4nv not supported in this architecture` on Ampere — leaving it out keeps A40 / A6000 /
3090 usable. `TorchCompile` costs 1-3 minutes of inductor compilation on every cold container
and recompiles per resolution. Both are speed optimizations on a pipeline that is correct
without them; add them back deliberately, and measure.

## Adding LoRAs without rebuilding

[`src/extra_model_paths.yaml`](src/extra_model_paths.yaml) maps `loras` to
`/runpod-volume/models/loras/`, and ComfyUI unions that with the baked `models/loras` — both
are visible at once. Attach a network volume, drop `.safetensors` files into `models/loras/` on
it from a cheap Pod, and reference them by filename in the workflow. No rebuild.

Two things to get right when stacking a LoRA on the 4-step distill:

- **Apply it to both experts, at different strengths.** A typical graph runs the lightx2v LoRA
  at ~0.4 on the high-noise expert and 1.0 on the low-noise one. Chain a second
  `LoraLoaderModelOnly` per expert and start conservative — high ≈ 0.3-0.5, low ≈ 0.6-0.8.
  High-noise LoRAs drive global composition and in image-to-video readily override the start
  frame, and a second LoRA at full strength on top of a distill LoRA tends to cost motion.
- **Check the licence.** The baked weights are Apache 2.0 and the node packs GPL-family or MIT,
  all fine commercially. Community LoRAs frequently ship with no licence at all, or a Civitai
  licence restricting commercial use.

## Verifying a workflow

A missing custom node shows up as a failed job on a paid GPU. Boot the image and diff your
workflow's classes against what ComfyUI registered:

```bash
curl -s localhost:8188/object_info | python3 -c '
import json, sys
have = set(json.load(sys.stdin))
want = {n["class_type"] for n in json.load(open("your_workflow.json")).values()}
print("MISSING:", want - have or "none")'
```

It must print `none`.

## Licence

AGPL-3.0, inherited from upstream. Serving this over a network is what §13 covers, so the
modified source has to be available — keeping this repository public satisfies that.
