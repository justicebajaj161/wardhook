# Wardhook developer tasks.
#
# Every target here has a one-to-one counterpart in .github/workflows/ci.yml,
# so `make check` locally is the same gate that runs on a pull request.

PACKAGES := wardhook-core wardhook-guardrails wardhook-observability wardhook-evals
PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help install lint fmt types test test-cov solo build clean check

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the dev environment with all four packages editable
	uv sync

lint: ## Lint and check formatting (does not modify files)
	uv run ruff check .
	uv run ruff format --check .

fmt: ## Autofix lint findings and format
	uv run ruff check --fix .
	uv run ruff format .

types: ## Type-check every package independently
	@set -e; for p in $(PACKAGES); do \
		echo "==> mypy $$p"; \
		uv run mypy packages/$$p/src; \
	done

test: ## Run every package's test suite
	uv run pytest

test-cov: ## Run tests with a coverage report
	uv run pytest --cov --cov-report=term-missing

solo: ## Prove each package installs and passes its tests entirely on its own
	@set -e; for p in $(PACKAGES); do \
		echo "==> standalone install: $$p"; \
		rm -rf .venv-solo; \
		uv venv --quiet .venv-solo; \
		VIRTUAL_ENV=.venv-solo uv pip install --quiet -e packages/$$p pytest; \
		VIRTUAL_ENV=.venv-solo uv run --no-project --active \
			pytest packages/$$p/tests -q -p no:cacheprovider; \
	done; \
	rm -rf .venv-solo; \
	echo "==> all four packages are independently installable"

build: ## Build wheels and sdists for all packages, then validate them
	@set -e; rm -rf dist; for p in $(PACKAGES); do \
		echo "==> build $$p"; \
		uv run $(PY) -m build --outdir dist packages/$$p; \
	done; \
	uv run twine check dist/*

clean: ## Remove build artifacts and caches
	rm -rf dist build .venv-solo .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

check: lint types test ## Run the full local gate (lint + types + tests)
