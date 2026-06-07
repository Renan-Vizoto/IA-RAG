#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_DIR="${EMBEDDING_MODEL_DIR:-${SCRIPT_DIR}/volumes/embeddings/paraphrase-multilingual-MiniLM-L12-v2}"
HF_MODEL="${EMBEDDING_MODEL_HF_ID:-sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2}"
HF_BASE="https://huggingface.co/${HF_MODEL}/resolve/main"

# Files required by sentence-transformers to load from a local folder.
MODEL_FILES=(
  config.json
  config_sentence_transformers.json
  modules.json
  sentence_bert_config.json
  special_tokens_map.json
  tokenizer.json
  tokenizer_config.json
  unigram.json
  model.safetensors
  1_Pooling/config.json
)

if [ -f "$MODEL_DIR/config.json" ] && [ -f "$MODEL_DIR/model.safetensors" ]; then
  echo "Embedding model already exists at $MODEL_DIR, skipping download."
else
  echo "Downloading embedding model to $MODEL_DIR ..."
  mkdir -p "$MODEL_DIR/1_Pooling"

  for file in "${MODEL_FILES[@]}"; do
    dest="$MODEL_DIR/$file"
    if [ -f "$dest" ]; then
      echo "  skip $file"
      continue
    fi
    echo "  -> $file"
    curl -L --fail --retry 3 --retry-delay 2 -o "$dest" "${HF_BASE}/${file}"
  done

  echo "Done: $MODEL_DIR"
fi

if [ $# -gt 0 ]; then
  exec "$@"
fi
