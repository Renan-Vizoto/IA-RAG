#!/bin/bash
# Verifica se o Ollama responde, o modelo padrão está registrado e o warmup concluiu.
MODEL="${OLLAMA_MODEL:-qwen3.5-2b-unsloth}"
WARMUP_ENABLED="${OLLAMA_WARMUP_ENABLED:-true}"

# Ollama lista modelos com sufixo :latest (ex.: qwen3.5-2b-unsloth:latest)
curl -sf http://127.0.0.1:11434/api/tags | grep -qF "$MODEL" || exit 1

if [ "$WARMUP_ENABLED" != "false" ]; then
  test -f /tmp/ollama-warmed || exit 1
fi
