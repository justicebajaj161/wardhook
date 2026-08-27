# Wardhook developer tasks.
#
# Every target here has a one-to-one counterpart in .github/workflows/ci.yml,
# so `make check` locally is the same gate that runs on a pull request.

# The four real packages: these have source, tests, and must each install
# alone. Used by `types`, `test`, and `solo`.
PACKAGES := wardhook-core wardhook-guardrails wardhook-observability wardhook-evals
# Plus the `wardhook` meta-package, which has no source and no tests but is
# published like the rest. Used by `build`.
ALL_PACKAGES := $(PACKAGES) wardhook
PY ?= python3

.DEFAULT_GOAL := help
.PHONY: help install lint fmt types test test-cov solo meta build clean check

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
	# Deliberately mirrors the `test` job in .github/workflows/ci.yml exactly:
	# the same extra packages, the same `-c /dev/null` (so the root pytest
	# config is NOT inherited), and the same sibling-leak assertion. Any drift
	# between the two means a failure only CI can see -- which has happened.
	@set -e; for p in $(PACKAGES); do \
		echo "==> standalone install: $$p"; \
		rm -rf .venv-solo; \
		uv venv --quiet .venv-solo; \
		VIRTUAL_ENV=.venv-solo uv pip install --quiet -e packages/$$p pytest pytest-cov httpx; \
		here=$${p#wardhook-}; \
		.venv-solo/bin/python -c "import sys; \
			siblings = {'core','guardrails','observability','evals'} - {'$$here'}; \
			leaked = [n for n in sorted(siblings) if __import__('importlib.util', fromlist=['x']).find_spec('wardhook.'+n)]; \
			sys.exit('FAIL: %s importable in a solo install' % leaked) if leaked else None"; \
		FORCE_COLOR=1 COLUMNS=80 .venv-solo/bin/python -m pytest packages/$$p/tests \
			-q -p no:cacheprovider --import-mode=importlib -c /dev/null; \
	done; \
	rm -rf .venv-solo; \
	echo "==> all four packages are independently installable"

build: ## Build wheels and sdists for all packages, then validate them
	@set -e; rm -rf dist; for p in $(ALL_PACKAGES); do \
		echo "==> build $$p"; \
		uv run $(PY) -m build --outdir dist packages/$$p; \
	done; \
	uv run twine check dist/*

clean: ## Remove build artifacts and caches
	rm -rf dist build .venv-solo .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
	find . -name '*.egg-info' -type d -prune -exec rm -rf {} +

meta: ## Prove `pip install wardhook` pulls in all four packages
	@set -e; rm -rf dist .venv-meta; \
	for p in $(ALL_PACKAGES); do \
		uv run $(PY) -m build --quiet --outdir dist packages/$$p >/dev/null; \
	done; \
	uv venv --quiet .venv-meta; \
	VIRTUAL_ENV=.venv-meta uv pip install --quiet --find-links dist wardhook; \
	.venv-meta/bin/python -c "from importlib.metadata import version; \
		import wardhook.core, wardhook.guardrails, wardhook.observability, wardhook.evals; \
		names = ['wardhook','wardhook-core','wardhook-guardrails','wardhook-observability','wardhook-evals']; \
		seen = {n: version(n) for n in names}; \
		assert len(set(seen.values())) == 1, 'versions out of lockstep: %s' % seen; \
		print('==> pip install wardhook -> ' + ', '.join(sorted(seen)) + ' all at ' + seen['wardhook'])"; \
	rm -rf .venv-meta

check: lint types test ## Run the full local gate (lint + types + tests)
