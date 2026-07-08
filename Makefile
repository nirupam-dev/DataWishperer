# ─────────────────────────────────────────────────────────────────────────────
# DataWhisperer — Makefile
#
# Provides a uniform command interface across all platforms.
# Run `make help` to see all available targets.
#
# Requirements: GNU Make, Docker (for containerised targets)
# ─────────────────────────────────────────────────────────────────────────────

.DEFAULT_GOAL := help
.PHONY: help install dev run test lint format typecheck clean \
        docker-up docker-up-gpu docker-down docker-build docker-logs \
        docker-reset health env setup

# ── Variables ────────────────────────────────────────────────────────────────
PYTHON      ?= python
PIP         ?= pip
STREAMLIT   ?= streamlit
COMPOSE     ?= docker compose
APP_PORT    ?= 8501
OLLAMA_MODEL?= qwen2.5:7b

# Colour codes for pretty output
CYAN  := \033[36m
GREEN := \033[32m
BOLD  := \033[1m
RESET := \033[0m

# ═════════════════════════════════════════════════════════════════════════════
# Help
# ═════════════════════════════════════════════════════════════════════════════

help: ## Show this help message
	@echo ""
	@echo "$(BOLD)DataWhisperer$(RESET) — Available Commands"
	@echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(CYAN)%-20s$(RESET) %s\n", $$1, $$2}'
	@echo ""

# ═════════════════════════════════════════════════════════════════════════════
# Setup & Installation
# ═════════════════════════════════════════════════════════════════════════════

env: ## Create .env from .env.example (safe — won't overwrite)
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo "$(GREEN)✓ Created .env from .env.example$(RESET)"; \
	else \
		echo "  .env already exists — skipping"; \
	fi

install: ## Install production dependencies
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements.txt

dev: ## Install all dependencies (production + dev/test)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -r requirements-dev.txt

setup: env install ## Full local setup (create .env + install deps)
	@echo "$(GREEN)✓ Setup complete. Run 'make run' to start.$(RESET)"

# ═════════════════════════════════════════════════════════════════════════════
# Development
# ═════════════════════════════════════════════════════════════════════════════

run: ## Run the Streamlit app locally
	$(STREAMLIT) run app.py --server.port=$(APP_PORT)

# ═════════════════════════════════════════════════════════════════════════════
# Quality Assurance
# ═════════════════════════════════════════════════════════════════════════════

test: ## Run test suite with coverage
	$(PYTHON) -m pytest tests/ -v --tb=short --cov=backend --cov=frontend \
		--cov-report=term-missing --cov-report=html

test-fast: ## Run tests excluding slow/integration markers
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow and not integration"

lint: ## Run linter (ruff)
	$(PYTHON) -m ruff check backend/ frontend/ tests/ app.py

format: ## Auto-format code (ruff)
	$(PYTHON) -m ruff format backend/ frontend/ tests/ app.py
	$(PYTHON) -m ruff check --fix backend/ frontend/ tests/ app.py

typecheck: ## Run static type checker (mypy)
	$(PYTHON) -m mypy backend/ frontend/

qa: lint typecheck test ## Run full QA pipeline (lint + types + tests)

# ═════════════════════════════════════════════════════════════════════════════
# Docker — Production Deployment
# ═════════════════════════════════════════════════════════════════════════════

docker-build: ## Build the Docker image
	$(COMPOSE) build --no-cache

docker-up: env ## Start all services (CPU mode)
	$(COMPOSE) up -d
	@echo ""
	@echo "$(GREEN)$(BOLD)DataWhisperer is starting...$(RESET)"
	@echo "  App:    http://localhost:$(APP_PORT)"
	@echo "  Ollama: http://localhost:11434"
	@echo ""
	@echo "  First run will pull model $(OLLAMA_MODEL) (~4GB)."
	@echo "  Run '$(CYAN)make docker-logs$(RESET)' to follow progress."

docker-up-gpu: env ## Start all services (GPU mode — NVIDIA)
	$(COMPOSE) --profile gpu up -d
	@echo ""
	@echo "$(GREEN)$(BOLD)DataWhisperer is starting (GPU mode)...$(RESET)"
	@echo "  App:    http://localhost:$(APP_PORT)"
	@echo "  Ollama: http://localhost:11434 (GPU accelerated)"

docker-down: ## Stop all services
	$(COMPOSE) down

docker-logs: ## Follow live logs from all services
	$(COMPOSE) logs -f

docker-reset: ## Stop services and remove all volumes (full reset)
	$(COMPOSE) down -v --remove-orphans
	@echo "$(GREEN)✓ All containers and volumes removed.$(RESET)"

health: ## Check health status of all running services
	@echo "$(BOLD)Service Health Status$(RESET)"
	@echo "━━━━━━━━━━━━━━━━━━━━━"
	@$(COMPOSE) ps --format "table {{.Name}}\t{{.Status}}\t{{.Ports}}" 2>/dev/null || \
		echo "  No services running. Run 'make docker-up' first."
	@echo ""
	@echo "Ollama API:"
	@curl -sf http://localhost:11434/api/tags > /dev/null 2>&1 && \
		echo "  $(GREEN)✓ Reachable$(RESET)" || echo "  ✗ Not reachable"
	@echo "Streamlit:"
	@curl -sf http://localhost:$(APP_PORT)/_stcore/health > /dev/null 2>&1 && \
		echo "  $(GREEN)✓ Healthy$(RESET)" || echo "  ✗ Not reachable"

# ═════════════════════════════════════════════════════════════════════════════
# Maintenance
# ═════════════════════════════════════════════════════════════════════════════

clean: ## Remove caches, build artifacts, and temporary files
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "$(GREEN)✓ Cleaned.$(RESET)"
