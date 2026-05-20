# ============================================================
# video-to-3d - Dockerfile  (Blackwell / RTX 50-series build)
# ============================================================
# Base: CUDA 12.8 + cuDNN on Ubuntu 22.04.
# CUDA 12.8 is the FIRST toolkit with Blackwell (sm_120) support.
# CUDA 12.4 does NOT support sm_120 - that is the root cause of the
# "no kernel image is available for execution on the device" error.
#
# Expect a LONG first build (flash-attn is compiled from source for
# sm_120). Subsequent builds are cached.
# ============================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Limit parallel compile jobs so the flash-attn source build doesn't OOM.
ENV MAX_JOBS=2
# Tell every source build to emit Blackwell (sm_120) kernels + PTX.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV FLASH_ATTN_CUDA_ARCHS="120"

# --- System packages ---
RUN apt-get update && apt-get install -y software-properties-common \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3.10-dev python3.10-venv python3-pip \
    git wget curl ffmpeg cmake build-essential ninja-build \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Virtual environment (isolated from system Python) ---
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel packaging psutil

# --- PyTorch 2.7.1 (CUDA 12.8) - ships sm_120 kernels for Blackwell ---
RUN pip install \
    torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128

# --- torch-scatter / pytorch3d: deliberately NOT installed ---
# pipeline/reconstruct.py monkey-patches sys.modules with pure-PyTorch
# replacements for both BEFORE importing AMB3R, so the real packages are
# never used. Installing them here would just add two more things to break.

# --- Core numeric stack (pinned first so nothing else moves it) ---
RUN pip install numpy==1.26.4 scipy==1.13.1

# --- AMB3R Python dependencies ---
# !!! xformers MUST be pinned to the release built for this exact torch
# version (0.0.31 -> torch 2.7.1). An unpinned `xformers` resolves to the
# latest release and SILENTLY upgrades torch, breaking the whole stack.
RUN pip install \
    opencv-python==4.10.0.84 \
    xformers==0.0.31 \
    open3d==0.18.0 \
    huggingface-hub==0.28.1 \
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

# --- Guard: abort the build LOUDLY if a dependency moved torch off 2.7.1 ---
RUN python -c "import torch; v=torch.__version__; assert v.startswith('2.7.1'), f'BUILD ABORTED: torch is {v}, a dependency upgraded it.'; print(f'[guard] torch OK: {v}')"

# --- flash-attn: built FROM SOURCE so the kernels include sm_120 ---
# A prebuilt cu124/torch2.6 wheel physically cannot run on Blackwell.
# Building against this CUDA 12.8 toolkit produces real sm_120 kernels.
# SLOW: ~20-40 min with MAX_JOBS=2. To skip the wait, replace this line
# with a prebuilt wheel from:
#   https://github.com/mjun0812/flash-attention-prebuild-wheels
# picking one tagged  cu128torch2.7-cp310 ... linux_x86_64
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# --- Verify the critical stack imports cleanly ---
# NOTE: torch.cuda.get_arch_list() CANNOT be used here. It returns [] unless a
# GPU is visible, and `docker build` has no GPU. We verify the CUDA build tag
# instead - cu128 wheels are the ones compiled with sm_120. The real sm_120
# check happens at RUN time, when the container has the GPU (see instructions).
RUN python -c "import torch; cu=torch.version.cuda; print('[verify] torch', torch.__version__, 'cuda', cu); assert cu and cu.startswith('12.8'), f'torch is NOT a cu128 build (cuda={cu})'" \
 && python -c "import flash_attn; print('[verify] flash_attn', flash_attn.__version__)" \
 && python -c "import xformers; print('[verify] xformers', xformers.__version__)"

# --- Clone AMB3R with all submodules ---
WORKDIR /opt
RUN git clone --recursive https://github.com/HengyiWang/amb3r.git

# --- Workspace ---
WORKDIR /workspace
COPY . /workspace/
VOLUME ["/workspace/checkpoints", "/workspace/inputs", "/workspace/outputs"]
ENV PYTHONPATH="/opt/amb3r:/opt/amb3r/thirdparty:${PYTHONPATH}"

CMD ["bash"]