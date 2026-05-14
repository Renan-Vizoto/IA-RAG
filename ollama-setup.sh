#!/bin/bash
set -e

apt-get update && apt-get install -y curl

GGUF_PATH="/root/.ollama/gemma-4-E2B-it-Q4_K_M.gguf"

if [ ! -f "$GGUF_PATH" ]; then
  echo "Downloading model..."
  mkdir -p /root/.ollama
  curl -L -o "$GGUF_PATH" \
    "https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/main/gemma-4-E2B-it-Q4_K_M.gguf"
else
  echo "Model already exists, skipping download."
fi

cat > /Modelfile << 'EOF'
FROM /root/.ollama/gemma-4-E2B-it-Q4_K_M.gguf

PARAMETER stop "<eos>"
PARAMETER stop "<|turn>"

PARAMETER temperature 1.0
PARAMETER top_p 0.95
PARAMETER top_k 64

TEMPLATE """<bos>{{ if .System }}<|turn>system
{{ .System }}
{{ end }}{{ range .Messages }}{{ if eq .Role "user" }}<|turn>user
{{ .Content }}
{{ else if eq .Role "assistant" }}<|turn>model
{{ .Content }}
{{ else if eq .Role "tool" }}<|tool_response>{{ .Content }} 
{{ end }}{{ end }}{{ if .Tools }}<|turn>model
{{ end }}"""
EOF

ollama serve &
OLLAMA_PID=$!
sleep 10

if ! ollama list | grep -q "gemma4-unsloth"; then
  echo "Creating model..."
  ollama create gemma4-unsloth -f /Modelfile
else
  echo "Model already registered, skipping create."
fi

wait $OLLAMA_PID