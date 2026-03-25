#!/bin/bash
set -e

apt-get update && apt-get install -y curl

GGUF_PATH="/root/.ollama/Qwen3.5-4B-UD-Q4_K_XL.gguf"

if [ ! -f "$GGUF_PATH" ]; then
  echo "Downloading model..."
  mkdir -p /root/.ollama
  curl -L -o "$GGUF_PATH" \
    "https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-UD-Q4_K_XL.gguf"
else
  echo "Model already exists, skipping download."
fi

cat > /Modelfile << 'EOF'
FROM /root/.ollama/Qwen3.5-4B-UD-Q4_K_XL.gguf

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|endoftext|>"

# Qwen3.5 thinking mode - set to 0 to disable /think tokens
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

ollama serve &
OLLAMA_PID=$!
sleep 10

if ! ollama list | grep -q "qwen3.5-unsloth"; then
  echo "Creating model..."
  ollama create qwen3.5-unsloth -f /Modelfile
else
  echo "Model already registered, skipping create."
fi

wait $OLLAMA_PID