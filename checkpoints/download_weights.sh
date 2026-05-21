#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# download_weights.sh
# Downloads the AMB3R checkpoint into checkpoints/.
#
# The semantic-segmentation model (Mask2Former) is baked into the Docker
# image at build time, so it is NOT downloaded here.
# ---------------------------------------------------------------------------
set -e

CKPT_DIR="$(cd "$(dirname "$0")" && pwd)"
AMB3R_PT="${CKPT_DIR}/amb3r.pt"
AMB3R_ID="14x0WW2rUE_he2hUEouP6ywSRnlJDeLel"

if [ -f "${AMB3R_PT}" ]; then
    echo "amb3r.pt already present at ${AMB3R_PT} — skipping."
    exit 0
fi

echo "Downloading AMB3R checkpoint (~1 GB) ..."
gdown "https://drive.google.com/uc?id=${AMB3R_ID}" -O "${AMB3R_PT}"

if [ -f "${AMB3R_PT}" ]; then
    echo "Done: ${AMB3R_PT}"
else
    echo "ERROR: download failed."
    echo "Download amb3r.pt manually from:"
    echo "  https://drive.google.com/file/d/${AMB3R_ID}/view"
    echo "and place it at: ${AMB3R_PT}"
    exit 1
fi