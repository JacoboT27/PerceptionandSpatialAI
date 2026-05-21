# ============================================================
# video-to-3d - Dockerfile  (multi-stage, Blackwell / sm_120)
# ============================================================
# Stage 1 (builder): full CUDA *devel* toolkit - compiles
#   flash-attn and torch-scatter for sm_120, installs all deps,
#   bakes in the segmentation model. This stage is DISCARDED.
# Stage 2 (runtime): slim CUDA *runtime* base - only the built
#   virtualenv, AMB3R, the model cache, and the few runtime
#   system libs. No compiler toolchain ships in the final image.
# ============================================================

# ============================================================
# Stage 1 - builder
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
# Don't keep pip's downloaded-wheel cache inside layers.
ENV PIP_NO_CACHE_DIR=1
# Limit parallel compile jobs so source builds don't OOM the host.
ENV MAX_JOBS=2
# Emit Blackwell (sm_120) kernels + PTX for every source build.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV FLASH_ATTN_CUDA_ARCHS="120"
# `docker build` has no GPU; FORCE_CUDA makes torch_scatter build CUDA
# kernels anyway (without it it silently builds CPU-only).
ENV FORCE_CUDA=1
ENV HF_HOME=/opt/hf-cache

# Build toolchain + Python.
RUN apt-get update && apt-get install -y software-properties-common \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl cmake build-essential ninja-build \
    && rm -rf /var/lib/apt/lists/*

# Virtual environment - the single artifact copied to the runtime stage.
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel packaging psutil

# PyTorch 2.7.1 (CUDA 12.8) - ships sm_120 kernels for Blackwell.
RUN pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# flash-attn from source for sm_120 (slowest layer - kept early).
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# Core numeric stack (pinned).
RUN pip install numpy==1.26.4 scipy==1.13.1

# torch-scatter from source for sm_120 (FORCE_CUDA above is essential).
RUN pip install git+https://github.com/rusty1s/pytorch_scatter.git --no-build-isolation

# AMB3R + semantic-stage dependencies.
# xformers MUST stay pinned (0.0.31 -> torch 2.7.1).
RUN pip install \
    opencv-python==4.10.0.84 \
    xformers==0.0.31 \
    open3d==0.18.0 \
    huggingface-hub==0.28.1 \
    transformers==4.46.0 \
    einops==0.8.0 \
    trimesh==4.4.9 \
    pillow==10.3.0 \
    gdown==5.2.0 \
    "imageio[ffmpeg]==2.35.1" \
    spconv-cu120==2.3.6 \
    timm==0.6.7 \
    omegaconf==2.3.0 \
    evo==1.31.1 \
    pytoml==0.1.21 \
    tensorboard==2.18.0 \
    scikit-image==0.24.0 \
    dill==0.3.8 \
    easydict==1.13 \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# Guard: abort if a dependency moved torch off 2.7.1.
RUN python -c "import torch; v=torch.__version__; assert v.startswith('2.7.1'), f'BUILD ABORTED: torch is {v}'; print(f'[guard] torch OK: {v}')"

# Verify the critical stack (no GPU needed at build time).
RUN python -c "import torch; cu=torch.version.cuda; print('[verify] torch', torch.__version__, 'cuda', cu); assert cu and cu.startswith('12.8'), f'torch is NOT a cu128 build (cuda={cu})'" \
 && python -c "import flash_attn; print('[verify] flash_attn', flash_attn.__version__)" \
 && python -c "import xformers; print('[verify] xformers', xformers.__version__)" \
 && python -c "import torch_scatter, os; d=os.path.dirname(torch_scatter.__file__); cu=[f for f in os.listdir(d) if 'cuda' in f and f.endswith('.so')]; print('[verify] torch_scatter CUDA ext:', cu); assert cu, 'torch_scatter built WITHOUT CUDA'"

# Bake the semantic-segmentation model into the HF cache.
RUN python -c "from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation; m='facebook/mask2former-swin-base-coco-panoptic'; AutoImageProcessor.from_pretrained(m); Mask2FormerForUniversalSegmentation.from_pretrained(m); print('[verify] segmentation model cached:', m)"

# Clone AMB3R, then strip its git history (not needed to run).
RUN git clone --recursive https://github.com/HengyiWang/amb3r.git /opt/amb3r \
 && rm -rf /opt/amb3r/.git


# ============================================================
# Stage 2 - runtime  (slim: no compiler toolchain)
# ============================================================
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Runtime system libs only:
#   python3.10  - the virtualenv's interpreter links to it
#   ffmpeg      - frame extraction
#   libgl1/glib - OpenCV + Open3D
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    ffmpeg \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the three build artifacts from the builder stage.
COPY --from=builder /opt/venv     /opt/venv
COPY --from=builder /opt/amb3r    /opt/amb3r
COPY --from=builder /opt/hf-cache /opt/hf-cache

ENV PATH="/opt/venv/bin:$PATH"
ENV HF_HOME=/opt/hf-cache
ENV PYTHONPATH="/opt/amb3r:/opt/amb3r/thirdparty"
# Reduce VRAM fragmentation on memory-constrained GPUs.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Confirm the copied venv resolves against the runtime python.
RUN python -c "import torch, flash_attn, torch_scatter, transformers; print('[verify] runtime stack OK -', torch.__version__)"

# --- Workspace (source code is volume-mounted at run time) ---
WORKDIR /workspace
COPY . /workspace/
VOLUME ["/workspace/checkpoints", "/workspace/inputs", "/workspace/outputs"]

CMD ["bash"]