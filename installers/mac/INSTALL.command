#!/bin/bash
set -euo pipefail

APP_NAME="AI Image Renamer"
INSTALL_DIR="$HOME/Applications/AI-Image-Renamer"

# Find Python 3
if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 not found. Install it from https://www.python.org/downloads/ and re-run."
  exit 1
fi

"$PYTHON_BIN" -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"
pip install -U pip

if [ -f "$INSTALL_DIR/requirements.txt" ]; then
  pip install -r "$INSTALL_DIR/requirements.txt"
else
  pip install requests pillow packaging
fi

cat > "$INSTALL_DIR/run_gui.sh" <<'RUN'
#!/bin/bash
set -e
APP_DIR="$HOME/Applications/AI-Image-Renamer"
source "$APP_DIR/venv/bin/activate"
exec python "$APP_DIR/src/ai_image_renamer.py"
RUN
chmod +x "$INSTALL_DIR/run_gui.sh"

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
