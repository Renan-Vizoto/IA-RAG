#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-qwen3.5-0.8b-unsloth}"

case "$MODEL" in
  gemma4-unsloth)
    exec bash /ollama-setup.sh
    ;;
  qwen3.5-0.8b-unsloth)
    exec bash /ollama-setup-qwen.sh
    ;;

  *)
    echo "Unknown OLLAMA_MODEL: $MODEL"
    echo "Supported: gemma4-unsloth, qwen3.5-0.8b-unsloth, qwen3.5-9b-unsloth"
    exit 1
    ;;
esac
