# IA-RAG — Docker Compose + model setup
#
# GPU profile (auto-detected when GPU=auto):
#   make up                          # detect nvidia → amd → cpu
#   make up GPU=nvidia               # force NVIDIA override
#   make up GPU=amd                  # force AMD/ROCm override
#   make up GPU=cpu                  # base compose only (CPU)
#
# Inference model:
#   make up OLLAMA_MODEL=qwen3.5-2b-unsloth

ROOT           := $(CURDIR)
VOLUMES        := $(ROOT)/volumes
OLLAMA_DIR     := $(VOLUMES)/ollama
EMBEDDING_DIR  := $(VOLUMES)/embeddings/paraphrase-multilingual-MiniLM-L12-v2
OLLAMA_MODEL   ?= qwen3.5-2b-unsloth
GPU            ?= auto
DATA_DIR       ?= notebooks/data
PIPELINE_FLAGS ?=
PYTHON         := $(ROOT)/.venv/bin/python

# auto | cpu | nvidia | amd
COMPOSE_GPU := $(shell \
	gpu="$(GPU)"; \
	if [ "$$gpu" != "auto" ]; then echo "$$gpu"; \
	elif command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then echo nvidia; \
	elif [ -e /dev/kfd ]; then echo amd; \
	else echo cpu; fi)

ifeq ($(COMPOSE_GPU),nvidia)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.nvidia.yml
else ifeq ($(COMPOSE_GPU),amd)
COMPOSE_FILES := -f docker-compose.yml -f docker-compose.amd.yml
else
COMPOSE_FILES := -f docker-compose.yml
endif

COMPOSE        := docker compose $(COMPOSE_FILES)

.PHONY: help gpu-info up up-cpu up-nvidia up-amd down restart build pull logs ps \
        setup setup-all setup-embeddings \
        setup-ollama setup-ollama-gemma setup-ollama-qwen setup-ollama-9b \
        train train-force train-local \
        test test-unit clean-volumes

help: ## Show available targets
	@echo "GPU profile: $(COMPOSE_GPU) (GPU=$(GPU))"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*##' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

gpu-info: ## Print detected/selected GPU profile and compose files
	@echo "GPU=$(GPU)"
	@echo "Profile: $(COMPOSE_GPU)"
	@echo "Compose: docker compose $(COMPOSE_FILES)"

# ---------------------------------------------------------------------------
# Docker Compose
# ---------------------------------------------------------------------------

up: ## Start all services (see gpu-info for profile)
	@echo "Using GPU profile: $(COMPOSE_GPU)"
	OLLAMA_MODEL=$(OLLAMA_MODEL) $(COMPOSE) up -d --build

up-cpu: ## Start stack without GPU overrides
	$(MAKE) up GPU=cpu

up-nvidia: ## Start stack with docker-compose.nvidia.yml
	$(MAKE) up GPU=nvidia

up-amd: ## Start stack with docker-compose.amd.yml (ROCm)
	$(MAKE) up GPU=amd

down: ## Stop all services
	$(COMPOSE) down

restart: down up ## Restart stack

build: ## Build API image only
	$(COMPOSE) build api

pull: ## Pull external images
	$(COMPOSE) pull

logs: ## Follow logs for all services
	$(COMPOSE) logs -f

ps: ## Show running containers
	$(COMPOSE) ps

# ---------------------------------------------------------------------------
# Model setup (host — writes to volumes/)
# ---------------------------------------------------------------------------

setup: setup-embeddings setup-ollama-gemma ## Download embedding + default Ollama GGUF

setup-all: setup setup-ollama-qwen ## Download embedding + gemma + qwen 2B

setup-embeddings: ## Download sentence-transformers model to volumes/embeddings/
	@bash embedding-setup.sh

setup-ollama-gemma: ## Download gemma-4-E2B-it-Q4_K_M.gguf
	@bash scripts/download-ollama-gguf.sh gemma

setup-ollama-qwen: ## Download Qwen3.5-2B-UD-Q4_K_XL.gguf
	@bash scripts/download-ollama-gguf.sh qwen

# ---------------------------------------------------------------------------
# Dutch Energy pipeline (bronze → silver → gold → XGBoost + MLflow)
# ---------------------------------------------------------------------------

train: ## Run full pipeline in Docker (requires: make up + CSVs in DATA_DIR)
	@test -d "$(ROOT)/$(DATA_DIR)" || { \
		echo "Dataset not found at $(DATA_DIR)."; \
		echo "Download from Kaggle and place CSVs in $(ROOT)/$(DATA_DIR)/"; \
		exit 1; \
	}
	@$(COMPOSE) ps --status running --services api 2>/dev/null | grep -q api || { \
		echo "API container is not running. Start the stack with: make up"; \
		exit 1; \
	}
	$(COMPOSE) exec -T api python -m app.pipeline.run_dutch_energy_pipeline \
		--data-dir $(DATA_DIR) $(PIPELINE_FLAGS)

train-force: ## Re-process silver/gold and retrain (PIPELINE_FLAGS=--force)
	$(MAKE) train PIPELINE_FLAGS=--force

# ---------------------------------------------------------------------------
# Dev / test
# ---------------------------------------------------------------------------

test: ## Run full pytest suite
	@$(ROOT)/.venv/bin/pytest tests/ -q

test-unit: ## Run unit tests (exclude integration)
	@$(ROOT)/.venv/bin/pytest tests/services tests/workers tests/api tests/infrastructure tests/core -q

# ---------------------------------------------------------------------------
# Maintenance
# ---------------------------------------------------------------------------

clean-volumes: ## Remove generated volume data (destructive)
	@echo "This removes $(VOLUMES)/ and named Docker volumes."
	@read -p "Type 'yes' to continue: " confirm && [ "$$confirm" = "yes" ]
	rm -rf "$(VOLUMES)/embeddings" "$(VOLUMES)/ollama" "$(VOLUMES)/milvus" "$(VOLUMES)/etcd"
	$(COMPOSE) down -v
