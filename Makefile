.PHONY: install lint typecheck test leakage smoke check reproduce

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

reproduce:          ## full study from the committed snapshot (ledger-idempotent)
	uv run quantlab data-report configs/base.yaml
	uv run quantlab study configs/study.yaml
	uv run quantlab validate-study
	uv run quantlab robustness configs/study.yaml
	uv run quantlab holdout configs/study.yaml --experiment H4-IV-TS12 \
		--unlock-holdout "reproduction of the pre-registered single holdout evaluation"
	uv run quantlab finalize configs/study.yaml
