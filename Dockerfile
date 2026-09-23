# syntax=docker/dockerfile:1
# Wan 2.2 image-to-video worker for RunPod serverless.
#
# One image, one job. Weights are baked in (~45 GiB) so a warm host loads them
# from local NVMe; the network volume is still read for extra LoRAs, see README.
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_PREFER_BINARY=1 \
    PYTHONUNBUFFERED=1

# ffmpeg is not optional: VHS_VideoCombine shells out to it to write the MP4.
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 \
      python3.12-venv \
      git \
      wget \
      curl \
      ffmpeg \
      libgl1 \
      libglib2.0-0 \
    && ln -sf /usr/bin/python3.12 /usr/bin/python \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# One venv, used by every later step and by start.sh. ComfyUI and the handler
# share it, so there is no workspace/launch split to get wrong.
RUN wget -qO- https://astral.sh/uv/install.sh | sh \
    && ln -s /root/.local/bin/uv /usr/local/bin/uv \
    && uv venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

RUN git clone --depth 1 --branch v0.34.0 https://github.com/comfyanonymous/ComfyUI.git /comfyui

ADD requirements.txt /requirements.txt

# torch is installed FIRST and pinned to +cu128: ComfyUI's requirements.txt asks
# for a bare `torch`, and PyPI now serves CUDA 13 builds that need driver >= 580.
# RunPod hosts advertise CUDA 12.8/12.9 (driver 570/575), where a cu13 torch
# fails CUDA init at startup. Installing it first satisfies the bare requirement.
#
# transformers/huggingface-hub are pinned in the same step because ComfyUI
# declares them with no upper bound, and 5.x / 1.x break it at import time.
RUN uv pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && uv pip install -r /comfyui/requirements.txt \
    && uv pip install "transformers>=4.50.3,<5" "huggingface-hub<1.0" \
    && uv pip install -r /requirements.txt

# Custom nodes the Wan workflow imports. Cloned directly — a network volume
# cannot supply nodes, and they have to be in the image.
#   KJNodes             - ScheduledCFGGuidance, ModelPassThrough, VRAM_Debug,
#                         DummyOut, INTConstant, FloatConstant
#   VideoHelperSuite    - VHS_VideoCombine, the MP4 writer
#   Frame-Interpolation - RIFE VFI, which doubles 16fps to 32fps
# EasyCache is not here: it is a core ComfyUI node as of 0.34.
RUN cd /comfyui/custom_nodes \
    && git clone --depth 1 https://github.com/kijai/ComfyUI-KJNodes.git \
    && git clone --depth 1 https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite.git \
    && git clone --depth 1 https://github.com/Fannovel16/ComfyUI-Frame-Interpolation.git \
    && for r in /comfyui/custom_nodes/*/requirements.txt; do uv pip install -r "$r"; done

# RIFE ships without weights; the node looks for them under its own ckpts/ dir.
RUN mkdir -p /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife \
    && wget -q -O /comfyui/custom_nodes/ComfyUI-Frame-Interpolation/ckpts/rife/rife49.pth \
         https://huggingface.co/hfmaster/models-moved/resolve/cab6dcee2fbb05e190dbb8f536fbdaa489031a14/rife/rife49.pth

# Start ComfyUI once on CPU so a custom node with conflicting dependencies fails
# the build here, instead of as a "server not reachable" error on a live worker.
RUN cd /comfyui && timeout 300 python main.py --quick-test-for-ci --cpu

# Weights, ~45 GiB. One download per layer so a failed fetch doesn't invalidate
# the others. The two 14B experts are a mixture-of-experts pair — high noise for
# the early steps and global composition, low noise for the rest — and the
# workflow needs both.
WORKDIR /comfyui
RUN mkdir -p models/diffusion_models models/text_encoders models/loras models/vae

ARG WAN22=https://huggingface.co/Comfy-Org/Wan_2.2_ComfyUI_Repackaged/resolve/main/split_files
ARG WAN21=https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files

# The two experts are the DaSiWa TrueVision v11 (SnatchKiss) fine-tune rather
# than the stock Comfy-Org checkpoints, pulled from Civitai with a build secret.
# TrueVision is NON-distilled, which is what this workflow wants: the 8-step
# sampling comes from the lightx2v LoRAs applied on top, further down. The
# distilled "Lightspeed" line would need those LoRAs removed.
#
# 18.12 GiB each — v11 is fp8-mixed and keeps more layers at bf16 than the
# stock fp8_scaled files did, which is where the extra size goes.
RUN --mount=type=secret,id=CIVITAI_TOKEN \
    curl -L -f -H "Authorization: Bearer $(cat /run/secrets/CIVITAI_TOKEN)" \
      "https://civitai.com/api/download/models/2959309?fileId=2841257" \
      -o models/diffusion_models/dasiwa_truevision_snatchkiss_v11_high.safetensors

RUN --mount=type=secret,id=CIVITAI_TOKEN \
    curl -L -f -H "Authorization: Bearer $(cat /run/secrets/CIVITAI_TOKEN)" \
      "https://civitai.com/api/download/models/2959520?fileId=2840642" \
      -o models/diffusion_models/dasiwa_truevision_snatchkiss_v11_low.safetensors

RUN wget -q -O models/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors \
      ${WAN21}/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

# lightx2v distill LoRAs: what makes 4-step sampling viable, one per expert.
RUN wget -q -O models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
      ${WAN22}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors \
    && wget -q -O models/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors \
      ${WAN22}/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors

# Wan 2.2 reuses the 2.1 VAE.
RUN wget -q -O models/vae/wan_2.1_vae.safetensors ${WAN22}/vae/wan_2.1_vae.safetensors

# Lets ComfyUI also read models from a mounted network volume, for LoRAs you
# want to swap without rebuilding 45 GiB.
ADD src/extra_model_paths.yaml ./

WORKDIR /
ADD src/start.sh src/network_volume.py handler.py ./
RUN chmod +x /start.sh

CMD ["/start.sh"]
