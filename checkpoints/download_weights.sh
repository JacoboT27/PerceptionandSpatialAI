#!/usr/bin/env bash
# ============================================================
# checkpoints/download_weights.sh
# Downloads the AMB3R pretrained checkpoint from Google Drive.
# Run this once before your first inference.
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT_PATH="$SCRIPT_DIR/amb3r.pt"

if [ -f "$CHECKPOINT_PATH" ]; then
    echo "[✓] Checkpoint already exists at: $CHECKPOINT_PATH"
    echo "    Delete it and re-run this script to force a fresh download."
    exit 0
fi

echo "[*] Downloading AMB3R checkpoint..."
echo "    Destination: $CHECKPOINT_PATH"
echo ""

# Try gdown first (Python-based, handles Google Drive auth well)
if command -v gdown &>/dev/null; then
    gdown "14x0WW2rUE_he2hUEouP6ywSRnlJDeLel" -O "$CHECKPOINT_PATH"
else
    echo "[!] gdown not found. Trying pip install gdown..."
    pip install gdown -q
    gdown "14x0WW2rUE_he2hUEouP6ywSRnlJDeLel" -O "$CHECKPOINT_PATH"
fi

if [ -f "$CHECKPOINT_PATH" ]; then
    SIZE=$(du -sh "$CHECKPOINT_PATH" | cut -f1)
    echo ""
    echo "[✓] Download complete. File size: $SIZE"
    echo "    Saved to: $CHECKPOINT_PATH"
else
    echo ""
    echo "[✗] Download failed."
    echo ""
    echo "    Please download manually from:"
    echo "    https://drive.google.com/file/d/14x0WW2rUE_he2hUEouP6ywSRnlJDeLel/view"
    echo ""
    echo "    Then place the file at: checkpoints/amb3r.pt"
    exit 1
fi