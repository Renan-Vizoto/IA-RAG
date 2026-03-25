#!/bin/bash
set -e

# Check if the model file exists
if [ ! -f /root/.ollama/Qwen3.5-9B-UD-Q4_K_XL.gguf ]; then
  echo "ERROR: Model file not found at /root/.ollama/Qwen3.5-9B-UD-Q4_K_XL.gguf"
  echo "Please ensure the file is present before running this script."
  exit 1
fi

echo "Model file found at /root/.ollama/Qwen3.5-9B-UD-Q4_K_XL.gguf"

cat > /Modelfile-9b << 'EOF'
FROM /root/.ollama/Qwen3.5-9B-UD-Q4_K_XL.gguf

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

ollama create qwen3.5-9b-unsloth -f /Modelfile-9b

wait $OLLAMA_PID