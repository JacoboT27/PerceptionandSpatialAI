# ============================================================
# video-to-3d — Dockerfile
# ============================================================
# Base: CUDA 12.1 + cuDNN 8 on Ubuntu 22.04
# Uses a Python virtual environment to fully isolate all
# dependencies from system packages — no version conflicts.
#
# First build takes ~30–45 min due to pytorch3d compilation.
# Subsequent builds are cached and near-instant.
# ============================================================

FROM nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

# --- System packages ---
ENV DEBIAN_FRONTEND=noninteractive
ENV MAX_JOBS=2
RUN apt-get update && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3.9-dev \
    python3.9-distutils \
    python3.9-venv \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    cmake \
    build-essential \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# --- Create and activate a virtual environment ---
# All pip installs go into /opt/venv — completely isolated from system Python.
# System packages at /usr/local/lib/python3.9/dist-packages/ are never touched.
RUN python3.9 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip setuptools wheel

# --- PyTorch 2.5 (CUDA 12.1) ---
RUN pip install \
    torch==2.5.0 \
    torchvision==0.20.0 \
    torchaudio==2.5.0 \
    --index-url https://download.pytorch.org/whl/cu121

# --- torch-scatter ---
RUN pip install torch-scatter==2.1.2 \
    -f https://data.pyg.org/whl/torch-2.5.0+cu121.html

# --- pytorch3d (builds from source — this is the slow step) ---
RUN pip install "git+https://github.com/facebookresearch/pytorch3d.git@V0.7.8" \
    --no-build-isolation

# --- flash-attn ---
RUN pip install flash-attn==2.7.3 --no-build-isolation

# --- Core numeric stack first (pinned so nothing downgrades them) ---
RUN pip install \
    numpy==1.26.4 \
    scipy==1.13.1

# --- AMB3R Python dependencies ---
RUN pip install \
    opencv-python==4.10.0.84 \
    xformers==0.0.28.post2 \
    open3d==0.18.0 \
    huggingface-hub==0.28.1 \
    einops==0.8.0 \
    trimesh==4.4.9 \
    pillow==10.3.0 \
    gdown==5.2.0 \
    "imageio[ffmpeg]==2.35.1" \
    spconv-cu121==2.3.8 \
    timm==0.6.7 \
    omegaconf==2.3.0 \
    evo==1.31.1 \
    pytoml==0.1.21 \
    tensorboard==2.18.0 \
    scikit-image==0.24.0 \
    dill==0.3.8 \
    easydict==1.13 \
    "utils3d @ git+https://github.com/EasternJournalist/utils3d.git@c5daf6f6c244d251f252102d09e9b7bcef791a38"

# --- Clone AMB3R with all submodules ---
WORKDIR /opt
RUN git clone --recursive https://github.com/HengyiWang/amb3r.git

# --- Set up working directory ---
WORKDIR /workspace
COPY . /workspace/

# Volumes mounted at runtime
VOLUME ["/workspace/checkpoints", "/workspace/inputs", "/workspace/outputs"]

# Python path — venv is already active via PATH, just add AMB3R sources
ENV PYTHONPATH="/opt/amb3r:/opt/amb3r/thirdparty:${PYTHONPATH}"

CMD ["bash"]