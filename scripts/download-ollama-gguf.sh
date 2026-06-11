#!/bin/bash
# Download a GGUF file into volumes/ollama (host-side, no Docker required).
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OLLAMA_DIR="${OLLAMA_DIR:-$ROOT_DIR/volumes/ollama}"

usage() {
  echo "Usage: $0 <gemma|qwen|qwen-9b>"
  exit 1
}

[ $# -eq 1 ] || usage

mkdir -p "$OLLAMA_DIR"

case "$1" in
  gemma)
    FILE="$OLLAMA_DIR/gemma-4-E2B-it-Q4_K_M.gguf"
    URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
    ;;
  qwen)
    FILE="$OLLAMA_DIR/Qwen3.5-2B-UD-Q4_K_XL.gguf"
    URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf"
    ;;
  qwen-9b)
    FILE="$OLLAMA_DIR/Qwen3.5-9B-UD-Q4_K_XL.gguf"
    echo "The 9B model must be placed manually at:"
    echo "  $FILE"
    if [ -f "$FILE" ]; then
      echo "File already present."
    else
      echo "Download from HuggingFace (unsloth/Qwen3.5-9B-GGUF) and copy it there."
      exit 1
    fi
    exit 0
    ;;
  *)
    usage
    ;;
esac

if [ -f "$FILE" ]; then
  echo "Already exists: $FILE"
  exit 0
fi

echo "Downloading to $FILE ..."
curl -L --progress-bar -o "$FILE" "$URL"
echo "Done: $FILE"
