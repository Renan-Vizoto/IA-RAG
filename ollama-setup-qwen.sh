#!/bin/bash
set -e

if [ -z "${REGISTER_ONLY:-}" ]; then
  apt-get update && apt-get install -y curl
fi

GGUF_PATH="/root/.ollama/Qwen3.5-2B-UD-Q4_K_XL.gguf"
HF_URL="https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-UD-Q4_K_XL.gguf"
MODEL_NAME="qwen3.5-2b-unsloth"
MODELFILE="/Modelfile-${MODEL_NAME}"

if [ ! -f "$GGUF_PATH" ]; then
  echo "Downloading Qwen3.5-2B model..."
  mkdir -p /root/.ollama
  curl -L -o "$GGUF_PATH" "$HF_URL"
else
  echo "Qwen model already exists, skipping download."
fi

# Qwen via GGUF precisa de TEMPLATE com tool_call; o renderer nativo não é aplicado ao FROM local.
cat > "$MODELFILE" << 'EOF'
FROM /root/.ollama/Qwen3.5-2B-UD-Q4_K_XL.gguf

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

PARAMETER temperature 0.6
PARAMETER top_p 0.95
PARAMETER top_k 20
PARAMETER min_p 0

TEMPLATE """{{- if or .System .Tools }}<|im_start|>system
{{- if .System }}
{{ .System }}
{{- end }}
{{- if .Tools }}

# Tools

You may call one or more functions to assist with the user query. You are provided with function signatures within <tools></tools> XML tags:

<tools>
{{- range .Tools }}
{"type": "function", "function": {{ .Function }}}
{{- end }}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>
{{- end }}<|im_end|>
{{ end }}
{{- range $i, $_ := .Messages }}
{{- $last := eq (len (slice $.Messages $i)) 1 }}
{{- if eq .Role "user" }}<|im_start|>user
{{ .Content }}<|im_end|>
{{ else if eq .Role "assistant" }}<|im_start|>assistant
{{- if .ToolCalls }}

{{- range .ToolCalls }}
<tool_call>
{"name": "{{ .Function.Name }}", "arguments": {{ .Function.Arguments }}}
</tool_call>
{{- end }}
{{- else }}
{{ .Content }}
{{- end }}<|im_end|>
{{ else if eq .Role "tool" }}<|im_start|>user
<tool_response>
{{ .Content }}
</tool_response><|im_end|>
{{ end }}
{{- if and $last (ne .Role "assistant") }}<|im_start|>assistant
{{ end }}
{{- end }}"""
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
