#!/bin/bash
set -euo pipefail

APP_NAME="AI Image Renamer"
INSTALL_DIR="$HOME/Applications/AI-Image-Renamer"

# Resolve the folder this script lives in after unzip
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PKG_DIR="$SCRIPT_DIR"   # files are packaged next to this script

# Find Python 3
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 not found. Install it from https://www.python.org/downloads/ and re-run."
  exit 1
fi

echo "Installing $APP_NAME..."

# Create venv
"$PYTHON_BIN" -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"
python -m pip install -U pip

# Ensure target layout
mkdir -p "$INSTALL_DIR/src"

# Copy application files from the package root into the install dir
if [ -f "$PKG_DIR/ai_image_renamer.py" ]; then
  cp -f "$PKG_DIR/ai_image_renamer.py" "$INSTALL_DIR/src/ai_image_renamer.py"
else
  echo "ERROR: ai_image_renamer.py not found next to INSTALL.command"
  exit 1
fi
[ -f "$PKG_DIR/auto_updater.py" ] && cp -f "$PKG_DIR/auto_updater.py" "$INSTALL_DIR/src/" || true
[ -f "$PKG_DIR/README.txt" ] && cp -f "$PKG_DIR/README.txt" "$INSTALL_DIR/" || true
[ -f "$PKG_DIR/requirements.txt" ] && cp -f "$PKG_DIR/requirements.txt" "$INSTALL_DIR/" || true

# Install deps
if [ -f "$INSTALL_DIR/requirements.txt" ]; then
  python -m pip install -r "$INSTALL_DIR/requirements.txt"
else
  python -m pip install requests pillow packaging
fi

# Launcher
cat > "$INSTALL_DIR/run_gui.sh" <<'RUN'
#!/bin/bash
set -e
APP_DIR="$HOME/Applications/AI-Image-Renamer"
source "$APP_DIR/venv/bin/activate"
exec python "$APP_DIR/src/ai_image_renamer.py"
RUN
chmod +x "$INSTALL_DIR/run_gui.sh"

# Simple .app wrapper that opens Terminal and runs the launcher
APP_BUNDLE="/Applications/$APP_NAME.app"
rm -rf "$APP_BUNDLE"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources"

cat > "$APP_BUNDLE/Contents/Info.plist" <<'PLIST'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
 "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>CFBundleName</key><string>AI Image Renamer</string>
  <key>CFBundleExecutable</key><string>run</string>
  <key>CFBundleIdentifier</key><string>com.sebastiank112.aiimagerenamer</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>LSMinimumSystemVersion</key><string>10.12</string>
</dict></plist>
PLIST

cat > "$APP_BUNDLE/Contents/MacOS/run" <<'WRAP'
#!/bin/bash
osascript <<'APPLESCRIPT'
tell application "Terminal"
  do script "/bin/bash -lc '$HOME/Applications/AI-Image-Renamer/run_gui.sh'"
  activate
end tell
APPLESCRIPT
WRAP
chmod +x "$APP_BUNDLE/Contents/MacOS/run"

echo "Installed to: $INSTALL_DIR"
echo "App created:   $APP_BUNDLE"
echo
echo "Next:"
echo " - Double click setup_api_key.command to add your OpenAI key"
echo " - Launch \"$APP_NAME\" from Applications"
