#!/bin/bash
set -e

MODEL="${1:-${OLLAMA_MODEL:-qwen3.5-2b-unsloth}}"
KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
WARMUP_ENABLED="${OLLAMA_WARMUP_ENABLED:-true}"
WARMUP_TIMEOUT="${OLLAMA_WARMUP_TIMEOUT:-360}"

if [ "$WARMUP_ENABLED" = "false" ]; then
  echo "[OLLAMA] Warmup desabilitado (OLLAMA_WARMUP_ENABLED=false), pulando."
  exit 0
fi

echo "[OLLAMA] Iniciando warmup do modelo: $MODEL (keep_alive=$KEEP_ALIVE)"
START=$(date +%s)

PAYLOAD=$(cat <<EOF
{
  "model": "$MODEL",
  "prompt": "warmup",
  "stream": false,
  "options": {"num_predict": 1},
  "keep_alive": "$KEEP_ALIVE"
}
EOF
)

if ! curl -sf --max-time "$WARMUP_TIMEOUT" \
  -H "Content-Type: application/json" \
  -d "$PAYLOAD" \
  "${OLLAMA_HOST}/api/generate" > /dev/null; then
  echo "[OLLAMA] ERRO: warmup falhou para modelo $MODEL" >&2
  exit 1
fi

ELAPSED=$(( $(date +%s) - START ))
touch /tmp/ollama-warmed
echo "[OLLAMA] Warmup concluído: $MODEL em ${ELAPSED}s"
