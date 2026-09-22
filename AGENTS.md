<!-- Do not edit or remove this section -->
<!-- This document exists for non-obvious, error-prone shortcomings in the codebase, the model, or the tooling that an agent cannot figure out by reading the code alone. No architecture overviews, file trees, build commands, or standard behavior. When you encounter something that belongs here, first consider whether a code change could eliminate it and suggest that to the user. Only document it here if it can't be reasonably fixed. -->

---

## Non-obvious constraints

- **No hot-reload**: handler.py, start.sh, and network_volume.py are `ADD`ed into the Docker image at build time (to `/`). Any change requires a full `docker build`.
- **Platform mismatch**: Always build with `--platform linux/amd64` for Runpod deployment. Omitting this on ARM hosts (Apple Silicon) produces images that silently fail on Runpod.
- **No linter or formatter configured**: Follow PEP 8 by convention; there are no pre-commit hooks or CI lint checks.
- **No ComfyUI-Manager**: ComfyUI is cloned at a pinned tag rather than installed through comfy-cli, so the Manager is not present. Custom nodes cannot be added at runtime — they must be baked into the Dockerfile.
- **Building locally is impractical**: the image is ~45 GB. CI (`.github/workflows/build-image.yml`) is the normal path, and it needs `easimon/maximize-build-space` to fit on a runner at all.
- **Network volume mount point**: Models on a network volume must match the directory structure in `src/extra_model_paths.yaml`. The volume is expected at `/runpod-volume` with a `comfyui/models/` subtree.

## Model type detection (for workflow parsing)

Node types map to model directories — this is ComfyUI domain knowledge not encoded in handler code:

- `UpscaleModelLoader` → `upscale_models`
- `VAELoader` → `vae`
- `UNETLoader`, `UnetLoaderGGUF`, `Hy3DModelLoader` → `diffusion_models`
- `DualCLIPLoader`, `TripleCLIPLoader` → `text_encoders`
- `LoraLoader` → `loras`

## Custom node compatibility

Some custom nodes have dependency conflicts that only surface at runtime:

- **ComfyUI-BrushNet**: Requires `diffusers>=0.29.0`, `accelerate>=0.29.0,<0.32.0`, and `peft>=0.7.0`. Without these exact ranges, you get silent import errors.
- **General pattern**: When a custom node fails with import errors, check its dependency chain and pin versions in the Dockerfile with `uv pip install`.
