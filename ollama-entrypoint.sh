#!/bin/bash
set -e

MODEL="${OLLAMA_MODEL:-qwen3.5-2b-unsloth}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"

if ! command -v curl >/dev/null 2>&1; then
  apt-get update && apt-get install -y curl
fi

ollama serve &
OLLAMA_PID=$!

echo "[OLLAMA] Aguardando servidor..."
for _ in $(seq 1 60); do
  if curl -sf "${OLLAMA_HOST}/api/tags" >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

export REGISTER_ONLY=1
bash /ollama-setup-qwen.sh
bash /ollama-setup.sh
if [ -f /ollama-setup-9b.sh ]; then
  bash /ollama-setup-9b.sh
fi

bash /ollama-warmup.sh "$MODEL"

wait $OLLAMA_PID
