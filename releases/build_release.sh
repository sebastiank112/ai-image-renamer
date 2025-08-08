#!/bin/bash
set -euo pipefail
VERSION="${1:-}"
[ -z "$VERSION" ] && { echo "Usage: $0 <version>"; exit 1; }

BUILD_DIR="temp_build_$VERSION"
PKG="AI_Image_Renamer_Mac_v$VERSION"

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR/$PKG"

cp src/ai_image_renamer.py "$BUILD_DIR/$PKG/ai_image_renamer.py"
[ -f src/auto_updater.py ] && cp src/auto_updater.py "$BUILD_DIR/$PKG/"
[ -d installers/mac ] && cp installers/mac/*.command "$BUILD_DIR/$PKG/" || true
[ -f README.md ] && cp README.md "$BUILD_DIR/$PKG/README.txt"

# Optional version bump if your main file has APP_VERSION = "x.y.z"
sed -i '' "s/APP_VERSION = \".*\"/APP_VERSION = \"$VERSION\"/" "$BUILD_DIR/$PKG/ai_image_renamer.py" || true

cd "$BUILD_DIR"
zip -r "../${PKG}.zip" "$PKG" >/dev/null
cd ..
rm -rf "$BUILD_DIR"
echo "Created ${PKG}.zip"
