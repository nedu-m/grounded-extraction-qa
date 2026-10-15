.PHONY: help install lint test test-cov report compare summary dashboard sqlqa serve ci clean

VENV := .venv
PY := $(VENV)/bin/python
GEQ := $(VENV)/bin/geq

help: ## Show available targets
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install: ## Create the venv and install dev + runtime deps
	uv venv --python 3.12 $(VENV)
	uv pip install --python $(PY) -e ".[dev]"

lint: ## Static analysis
	$(VENV)/bin/ruff check src tests
	$(VENV)/bin/ruff format --check src tests

test: ## Run the test suite
	$(PY) -m pytest

test-cov: ## Run the test suite with a coverage gate
	$(PY) -m pytest --cov=geq --cov-report=term-missing --cov-fail-under=85

report: ## Score every extractor and control against the golden set
	$(GEQ) run --corpus data/golden --extractor rules:v1 --out reports/rules-v1
	$(GEQ) run --corpus data/golden --extractor rules:v2 --out reports/rules-v2
	$(GEQ) run --corpus data/golden --extractor rules:v3 --out reports/rules-v3
	$(GEQ) run --corpus data/golden --extractor adversarial:uncited --out reports/adversarial-uncited
	$(GEQ) run --corpus data/golden --extractor adversarial:fabricating --out reports/adversarial-fabricating
	$(GEQ) summary --reports reports

compare: ## Regression-gate v3 against v2 (non-zero exit on any regression)
	$(GEQ) compare --baseline reports/rules-v2.json --candidate reports/rules-v3.json

summary: ## Cross-revision table from everything in reports/
	$(GEQ) summary --reports reports --out reports/summary.md

dashboard: ## Render the dashboard to reports/dashboard.html
	$(PY) -c "from geq.web import build_dashboard, render_dashboard; \
	          from pathlib import Path; \
	          Path('reports').mkdir(exist_ok=True); \
	          Path('reports/dashboard.html').write_text(render_dashboard(build_dashboard()))"
	@echo "  wrote reports/dashboard.html"

sqlqa: ## Run the SQL data-quality checks against the seeded database
	$(GEQ) sqlqa --schema sql/schema.sql --seed sql/seed.sql

serve: ## Run the QA service locally
	$(VENV)/bin/uvicorn geq.api:app --reload --port 8000

ci: lint test test-cov report compare sqlqa ## Everything CI runs

clean: ## Remove generated artefacts
	rm -rf reports/*.json reports/*.md reports/*.html .pytest_cache .ruff_cache .coverage htmlcov
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
