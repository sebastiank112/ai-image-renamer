#!/bin/bash
set -euo pipefail
ENV_FILE="$HOME/.ai_renamer.env"

API_KEY="$(osascript -e 'display dialog "Enter your OpenAI API key (starts with sk-)" default answer "" with title "AI Image Renamer" buttons {"OK"} default button "OK" with icon note' -e 'text returned of result' 2>/dev/null || true)"
if [[ -z "${API_KEY:-}" ]]; then echo "No key entered. Aborting."; exit 1; fi
if [[ "$API_KEY" != sk-* ]]; then
  osascript -e 'display alert "Invalid key" message "It should start with sk-. Try again." as warning'
  exit 1
fi

printf "OPENAI_API_KEY=%s\n" "$API_KEY" > "$ENV_FILE"
chmod 600 "$ENV_FILE"
osascript -e 'display notification "API key saved" with title "AI Image Renamer"'
echo "Saved to $ENV_FILE"
