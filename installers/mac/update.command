#!/bin/bash
set -euo pipefail
REPO="sebastiank112/ai-image-renamer"
INSTALL_DIR="$HOME/Applications/AI-Image-Renamer"
TMP_DIR="$(mktemp -d)"

echo "Checking latest release..."
TAG=$(python3 - <<'PY'
import json, urllib.request
with urllib.request.urlopen("https://api.github.com/repos/sebastiank112/ai-image-renamer/releases/latest") as r:
    print((json.load(r).get("tag_name","")).lstrip("v"))
PY
)

if [ -z "$TAG" ]; then echo "Could not determine latest version."; exit 1; fi
echo "Latest: v$TAG"

ZIP_URL="https://github.com/${REPO}/releases/download/v${TAG}/AI_Image_Renamer_Mac_v${TAG}.zip"
ZIP_PATH="$TMP_DIR/release.zip"

echo "Downloading $ZIP_URL"
curl -L -o "$ZIP_PATH" "$ZIP_URL"

echo "Unpacking..."
unzip -q "$ZIP_PATH" -d "$TMP_DIR"
PKG_DIR="$TMP_DIR/AI_Image_Renamer_Mac_v${TAG}"

mkdir -p "$INSTALL_DIR/src"
cp -f "$PKG_DIR/ai_image_renamer.py" "$INSTALL_DIR/src/"
[ -f "$PKG_DIR/auto_updater.py" ] && cp -f "$PKG_DIR/auto_updater.py" "$INSTALL_DIR/src/" || true
[ -f "$PKG_DIR/README.txt" ] && cp -f "$PKG_DIR/README.txt" "$INSTALL_DIR/" || true
[ -f "$PKG_DIR/requirements.txt" ] && cp -f "$PKG_DIR/requirements.txt" "$INSTALL_DIR/" || true

if [ -d "$INSTALL_DIR/venv" ]; then
  source "$INSTALL_DIR/venv/bin/activate"
  if [ -f "$INSTALL_DIR/requirements.txt" ]; then
    pip install -r "$INSTALL_DIR/requirements.txt"
  else
    pip install -U requests pillow packaging
  fi
fi

osascript -e 'display notification "AI Image Renamer updated" with title "Update complete"'
echo "Update complete."
