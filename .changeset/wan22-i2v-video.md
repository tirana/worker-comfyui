---
"worker-comfyui": minor
---

Return video and audio output, and add a `wan2.2-i2v` image variant.

The handler only ever collected `node_output["images"]`. Video nodes report saved files under a different key — `VHS_VideoCombine` uses `gifs` whatever container it actually wrote, ComfyUI's native video nodes use `videos` — so any workflow ending in a video node completed successfully and returned an empty `images` list, with the clip discarded and only a "produced unhandled output keys" warning to show for it. Every one of those keys carries the same `{filename, subfolder, type}` entries and is fetched identically through `/view`, so they are now flattened into one list via `MEDIA_KEYS` and the existing fetch-and-encode path handles them unchanged: it already derives the extension with `os.path.splitext(filename)[1]`, so `.mp4` comes back correctly. Output still arrives under `output.images`, which keeps existing clients working. This is a strict superset of the previous behaviour.

Adds `MODEL_TYPE=wan2.2-i2v`, which bakes Wan 2.2 I2V A14B (both mixture-of-experts halves, the lightx2v 4-step distill LoRAs, the umt5 text encoder and the 2.1 VAE — ~35 GiB, all Apache 2.0) plus the three custom node packs a Wan image-to-video workflow needs: kjnodes, videohelpersuite and frame-interpolation with RIFE weights. Node requirements are re-installed into `/opt/venv` after `comfy-node-install`, because comfy-cli installs them into its own workspace venv while `start.sh` launches ComfyUI with `/opt/venv`'s python — the mismatch documented in the base stage, which otherwise surfaces as a runtime import failure rather than a build error. The base stage's smoke test is re-run afterwards so a node with conflicting dependencies fails the build instead of a live worker. SageAttention and TorchCompile are deliberately omitted; SageAttention's kernels require sm_89+, which would restrict the image to Ada.

The variant is excluded from the `default` bake group so a plain `docker buildx bake` doesn't pull 35 GiB, and ships with a manual GHCR workflow that reclaims enough runner disk to build it. See `docs/wan2.2-i2v.md`.
