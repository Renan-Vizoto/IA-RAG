#!/bin/bash
set -e

if [ -z "${REGISTER_ONLY:-}" ]; then
  apt-get update && apt-get install -y curl
fi

GGUF_PATH="/root/.ollama/gemma-4-E2B-it-Q4_K_M.gguf"
HF_URL="https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
MODEL_NAME="gemma4-unsloth"
MODELFILE="/Modelfile-${MODEL_NAME}"

if [ ! -f "$GGUF_PATH" ]; then
  echo "Downloading Gemma 4 model..."
  mkdir -p /root/.ollama
  curl -L -o "$GGUF_PATH" "$HF_URL"
else
  echo "Gemma model already exists, skipping download."
fi

# Sem TEMPLATE: o Ollama aplica o Gemma4Renderer nativo (tools, tool_call, tool_response).
cat > "$MODELFILE" << 'EOF'
FROM /root/.ollama/gemma-4-E2B-it-Q4_K_M.gguf

PARAMETER stop "<eos>"
PARAMETER stop "<|turn>"

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64
EOF

register_model() {
  ollama rm -f "$MODEL_NAME" 2>/dev/null || true
  echo "Creating model $MODEL_NAME..."
  ollama create "$MODEL_NAME" -f "$MODELFILE"
}

if [ -n "${REGISTER_ONLY:-}" ]; then
  register_model
  echo "[OLLAMA] Modelo $MODEL_NAME registrado."
  exit 0
fi

ollama serve &
OLLAMA_PID=$!
sleep 10

register_model

bash /ollama-warmup.sh "$MODEL_NAME"

wait $OLLAMA_PID
