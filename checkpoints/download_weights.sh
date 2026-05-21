#!/usr/bin/env bash
# ============================================================
# checkpoints/download_weights.sh
# Downloads all required model checkpoints.
# Run this once before your first inference.
# ============================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---- Helper ------------------------------------------------
check_exists() {
    if [ -f "$1" ]; then
        echo "[✓] Already exists: $1"
        return 0
    fi
    return 1
}

# ------------------------------------------------------------ #
# 1. AMB3R
# ------------------------------------------------------------ #
AMB3R_PATH="$SCRIPT_DIR/amb3r.pt"
if ! check_exists "$AMB3R_PATH"; then
    echo "[*] Downloading AMB3R checkpoint (~1 GB)..."
    gdown "14x0WW2rUE_he2hUEouP6ywSRnlJDeLel" -O "$AMB3R_PATH"
    echo "[✓] AMB3R saved to: $AMB3R_PATH"
fi

# ------------------------------------------------------------ #
# 2. SAM ViT-B
# ------------------------------------------------------------ #
SAM_PATH="$SCRIPT_DIR/sam_vit_b.pth"
if ! check_exists "$SAM_PATH"; then
    echo "[*] Downloading SAM ViT-B checkpoint (~375 MB)..."
    wget -q --show-progress \
        "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth" \
        -O "$SAM_PATH"
    echo "[✓] SAM saved to: $SAM_PATH"
fi

# ------------------------------------------------------------ #
# 3. RAM++ (Recognize Anything Model Plus)
# ------------------------------------------------------------ #
RAM_PATH="$SCRIPT_DIR/ram_plus.pth"
if ! check_exists "$RAM_PATH"; then
    echo "[*] Downloading RAM++ checkpoint (~1.5 GB)..."
    wget -q --show-progress \
        "https://huggingface.co/xinyu1205/recognize-anything-plus-model/resolve/main/ram_plus_swin_large_14m.pth" \
        -O "$RAM_PATH"
    echo "[✓] RAM++ saved to: $RAM_PATH"
fi

echo ""
echo "[✓] All checkpoints ready."
echo "    AMB3R : $AMB3R_PATH"
echo "    SAM   : $SAM_PATH"
echo "    RAM++ : $RAM_PATH"