# quantlab

Framework de investigación cuantitativa para carteras multi-activo de horizonte
semanas–meses. Su objetivo es **no engañarse**: integridad temporal por
construcción, holdout sellado, conteo de pruebas para corregir por multiple
testing y costes siempre incluidos.

> Estado: **Fase 1 — infraestructura** completa. Aún no hay datos reales ni
> señales; el único experimento ejecutable es un smoke test sobre datos sintéticos.

## Mandato (fijado en Fase 0, antes de mirar datos)

| Parámetro      | Valor                                                        |
|----------------|--------------------------------------------------------------|
| Universo       | 18 ETFs multi-activo (acciones, bonos, crédito, real assets, cash) — `configs/base.yaml` |
| Datos          | Solo gratuitos (yfinance, FRED) vía adaptadores intercambiables |
| Rebalanceo     | Mensual; señal al cierre de `t`, ejecución al cierre de `t+1` |
| Cartera        | Long-only, sin apalancamiento, máx. 20% por posición, AUM 1M USD |
| Benchmark      | 60/40 SPY/AGG, rebalanceo mensual, con los mismos costes     |
| Objetivo       | Vol. ~10% anual; superar al benchmark en Sharpe neto de costes |
| Splits         | Warm-up 2002-07 → research 2005-01..2022-12 → **holdout 2023-01+ (sellado)** |

## Reproducir

```bash
uv sync --frozen                         # entorno exacto desde uv.lock
uv run quantlab run configs/smoke.yaml   # experimento end-to-end (sintético, offline)
make check                               # lint + mypy strict + tests + smoke
```

Un resultado queda identificado por **(commit git, hash de config, id de snapshot de datos)**;
los tres se guardan en `runs/<run_id>/run.json` y en el ledger.

### CLI

| Comando | Qué hace |
|---|---|
| `quantlab validate CONFIG` | Valida el esquema y resuelve los componentes; imprime el hash de config |
| `quantlab snapshot CONFIG` | Descarga datos y los congela como snapshot direccionado por contenido |
| `quantlab run CONFIG` | Corre un experimento en etapa *research* (nunca ve el holdout) |
| `quantlab run CONFIG --stage final --unlock-holdout "motivo"` | Evaluación final; el desbloqueo queda registrado |
| `quantlab trials [--hypothesis H]` | Nº de configuraciones probadas (input del Deflated Sharpe / PBO) |

## Arquitectura

```
DataSource → DataValidator → Factor → (Model) → SignalGenerator →
PortfolioConstructor → RiskManager → Backtester(+CostModel) → Evaluator → Reporter
```

- `src/quantlab/contracts.py` — interfaces abstractas de cada etapa.
- `src/quantlab/registry.py` — los componentes se registran por nombre; la config elige la implementación.
- `src/quantlab/config.py` — config declarativa pydantic (estricta, inmutable, con hash); herencia con `extends:`.
- `src/quantlab/pipeline.py` — orquestador: en cada fecha de decisión entrega a los componentes **solo** `data.asof(t)`.
- `src/quantlab/splits.py` — `HoldoutGuard`: el holdout solo se abre con un desbloqueo registrado.
- `src/quantlab/tracking.py` — tracking de corridas (`runs/`) + **ledger de pruebas append-only** (`research/trials.jsonl`).
- `src/quantlab/data/store.py` — snapshots de datos inmutables con checksums SHA-256.
- `src/quantlab/testing/leakage.py` — `assert_causal`: test adversarial de look-ahead obligatorio para cada factor.
- `src/quantlab/components/` — implementaciones: fuente sintética, validador, equal-weight, pesos fijos, costes spread+slippage+impacto √, backtester diario, evaluador.

Decisiones de diseño razonadas en [`docs/adr/`](docs/adr). Registro de hipótesis y
veredictos en [`docs/research_log.md`](docs/research_log.md); plantilla de
pre-registro en [`docs/hypotheses/TEMPLATE.md`](docs/hypotheses/TEMPLATE.md).

## Garantías y cómo se verifican

| Garantía | Mecanismo | Test |
|---|---|---|
| Sin look-ahead por el canal de datos | El orquestador trunca con `asof(t)`; `DecisionContext` rechaza historia > `as_of` | `test_components_never_see_the_future` |
| Sin look-ahead dentro de un factor | `assert_causal` (truncado + futuro aleatorizado) | `test_leakage.py` (detecta shift(-1), ventanas centradas, z-score full-sample, bfill, niveles ajustados…) |
| Holdout intacto | Vista research cortada en `research_end`; final requiere desbloqueo ledgered | `test_research_results_independent_of_holdout_data`, `test_final_stage_requires_unlock` |
| No operar al cierre de la señal | `execution_lag_days ≥ 1` validado en config | `test_signal_cannot_trade_on_its_own_close` |
| Conteo de pruebas | Toda corrida (incl. fallidas) va al ledger append-only; hook + CI lo verifican | `test_ledger_counts_distinct_successful_configs` |
| Datos inmutables | Snapshot direccionado por contenido + checksum al cargar | `test_snapshot_tampering_detected` |
| Mandato | `check_constraints` siempre activo tras construir/ajustar | `test_constraint_check_rejects_breaches` |
| Reproducibilidad | Seeds, `uv.lock`, hash de config, commit, fingerprint del entorno | `test_run_is_deterministic` |

## Limitaciones conocidas (honestas)

- **Survivorship en ETFs:** yfinance no incluye ETFs liquidados. Mitigación: universo fijado ex-ante y elegibilidad por fecha de inicio; residual documentado.
- **Ajustes vendor:** el "adjusted close" está retro-ajustado; solo los ratios son point-in-time. Regla: los factores usan retornos/ratios, nunca niveles (lo detecta `assert_causal`).
- **Cash al 0%** hasta conectar T-bills (FRED) en Fase 2: infla Sharpe absolutos; las comparaciones contra benchmark no se ven afectadas.
- **Muestra corta:** ~18 años de research ≈ 216 rebalanceos mensuales → poca potencia estadística.

## Roadmap

1. ✅ Infraestructura · 2. Datos reales (yfinance/FRED, validación, snapshot) · 3. Hipótesis pre-registradas + factores ·
4. Construcción de cartera y riesgo · 5. Walk-forward/CPCV, DSR, PBO · 6. Robustez + evaluación única en holdout · 7. Informe al comité
