.PHONY: install lint typecheck test leakage smoke check

install:            ## exact, locked environment + git hooks
	uv sync --frozen
	uv run pre-commit install

lint:
	uv run ruff format --check .
	uv run ruff check .

typecheck:
	uv run mypy

test:
	uv run pytest --cov=quantlab --cov-report=term-missing

leakage:            ## only the adversarial look-ahead tests
	uv run pytest -m leakage -v

smoke:              ## offline end-to-end run on synthetic data
	uv run quantlab run configs/smoke.yaml

check: lint typecheck test smoke
