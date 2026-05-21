# ============================================================
# video-to-3d - Dockerfile  (Blackwell / RTX 50-series build)
# ============================================================
# Base: CUDA 12.8 + cuDNN on Ubuntu 22.04.
# CUDA 12.8 is the first toolkit with Blackwell (sm_120) support;
# CUDA 12.4 does NOT support sm_120.
#
# Build-order note: flash-attn (the slowest layer) is compiled
# right after PyTorch, so editing later dependencies does NOT
# invalidate its cache. Expect a long FIRST build only.
# ============================================================

FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# Limit parallel compile jobs so source builds don't OOM the host.
ENV MAX_JOBS=2
# Emit Blackwell (sm_120) kernels + PTX for every source build.
ENV TORCH_CUDA_ARCH_LIST="12.0+PTX"
ENV FLASH_ATTN_CUDA_ARCHS="120"
# `docker build` has NO GPU, so torch.cuda.is_available() is False at build
# time. Without FORCE_CUDA, torch_scatter (and similar PyG packages) detect
# "no GPU" and silently build CPU-ONLY -> "Not compiled with CUDA support"
# at run time. FORCE_CUDA=1 makes them compile CUDA kernels regardless.
ENV FORCE_CUDA=1

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

# --- flash-attn: built FROM SOURCE for sm_120, right after torch ---
# Placed early (slowest + most stable layer) so editing later dependencies
# does not invalidate its cache. ~20-40 min with MAX_JOBS=2. To skip the
# wait, swap for a prebuilt wheel from
#   https://github.com/mjun0812/flash-attention-prebuild-wheels
# picking one tagged  cu128torch2.7-cp310 ... linux_x86_64
RUN pip install flash-attn==2.7.4.post1 --no-build-isolation

# --- Core numeric stack (pinned) ---
RUN pip install numpy==1.26.4 scipy==1.13.1

# --- torch-scatter: built FROM SOURCE against this torch, for sm_120 ---
# AMB3R's PTV3 backend calls torch_scatter.segment_csr. FORCE_CUDA=1 (set
# above) is ESSENTIAL here - without it this builds CPU-only and fails at
# run time with "Not compiled with CUDA support".
RUN pip install git+https://github.com/rusty1s/pytorch_scatter.git --no-build-isolation

# --- pytorch3d: deliberately NOT installed ---
# pytorch3d genuinely fails to compile for this stack. AMB3R uses only
# knn_points / knn_gather from it, which reconstruct.py monkey-patches with
# a complete pure-PyTorch implementation. That single patch stays.

# --- AMB3R Python dependencies ---
# !!! xformers MUST stay pinned (0.0.31 -> torch 2.7.1). An unpinned
# xformers silently upgrades torch and breaks flash-attn / torch_scatter.
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

# --- Guard: abort the build if a dependency moved torch off 2.7.1 ---
RUN python -c "import torch; v=torch.__version__; assert v.startswith('2.7.1'), f'BUILD ABORTED: torch is {v}'; print(f'[guard] torch OK: {v}')"

# --- Verify the critical stack at build time (no GPU needed) ---
# get_arch_list() can't be used here (returns [] without a GPU). Instead we
# confirm the cu128 build tag and that torch_scatter compiled a CUDA ext.
RUN python -c "import torch; cu=torch.version.cuda; print('[verify] torch', torch.__version__, 'cuda', cu); assert cu and cu.startswith('12.8'), f'torch is NOT a cu128 build (cuda={cu})'" \
 && python -c "import flash_attn; print('[verify] flash_attn', flash_attn.__version__)" \
 && python -c "import xformers; print('[verify] xformers', xformers.__version__)" \
 && python -c "import torch_scatter, os; d=os.path.dirname(torch_scatter.__file__); cu=[f for f in os.listdir(d) if 'cuda' in f and f.endswith('.so')]; print('[verify] torch_scatter CUDA ext:', cu); assert cu, 'torch_scatter built WITHOUT CUDA (FORCE_CUDA not set?)'"

# --- Semantic labelling dependencies (RAM++ + SAM + CLIP) ---
# Added after the verify block so editing these never invalidates
# the slow flash-attn / torch_scatter / xformers layers above.
#
# RAM++ : open-vocabulary automatic image tagger (no prompts needed)
# SAM   : Segment Anything Model — automatic mask generation
# CLIP  : assigns RAM tags to SAM masks by image-text similarity
RUN pip install \
    segment-anything \
    ftfy \
    regex \
    fairscale \
    "transformers==4.35.2" \
    "git+https://github.com/openai/CLIP.git" \
    "git+https://github.com/xinyu1205/recognize-anything.git"

# --- Guard: confirm semantic deps didn't move torch off 2.7.1 ---
RUN python -c "import torch; v=torch.__version__; assert v.startswith('2.7.1'), f'SEMANTIC DEPS ABORTED: torch moved to {v}'; print(f'[guard] torch still OK after semantic deps: {v}')"

# --- Clone AMB3R with all submodules ---
WORKDIR /opt
RUN git clone --recursive https://github.com/HengyiWang/amb3r.git

# --- Workspace ---
WORKDIR /workspace
COPY . /workspace/
VOLUME ["/workspace/checkpoints", "/workspace/inputs", "/workspace/outputs"]
ENV PYTHONPATH="/opt/amb3r:/opt/amb3r/thirdparty:${PYTHONPATH}"

# --- Reduce VRAM fragmentation (helps on memory-constrained GPUs) ---
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD ["bash"]