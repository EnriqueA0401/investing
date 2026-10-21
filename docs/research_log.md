# Research log

Registro cronológico de decisiones, hipótesis y veredictos, **incluidos los fallos**.
Cada entrada referencia los commits y, cuando aplique, los `run_id` / hashes del ledger.

---

## 2026-10-21 — Fase 0: alcance

Decisiones tomadas *antes de mirar ningún dato*:
- Universo: 18 ETFs multi-activo listados en EE. UU. (ver `configs/base.yaml`). Se
  eligieron por cobertura de clases de activo y liquidez, no por su rendimiento.
- Solo datos gratuitos. Rebalanceo mensual. Long-only, sin apalancamiento, máximo 20%
  por posición, AUM de 1M USD.
- Benchmark 60/40 SPY/AGG. Objetivo: volatilidad ~10% y Sharpe neto mayor que el benchmark.
- Holdout sellado desde 2023-01-01.

Riesgos aceptados de entrada:
- Muestra corta (~216 meses en research).
- Survivorship residual en ETFs.
- Cash al 0% hasta la Fase 2.

## 2026-10-21 — Fase 1: infraestructura

- Construido el framework: contratos, registro, config, orquestador point-in-time,
  holdout guard, ledger, snapshots, costes y backtester.
- **Hallazgo durante el desarrollo:** la primera versión de `assert_causal` solo
  probaba cortes en el último 70% de la muestra y **no detectaba un `bfill`**
  que filtraba datos al inicio de las series. Se corrigió repartiendo los cortes
  por toda la muestra y se añadió un test con un activo de listado tardío. Lección:
  los tests de fuga también hay que probarlos con fugas conocidas
  (`tests/test_leakage.py::LEAKY`).
- **Hallazgo:** la primera corrida del smoke test falló porque el benchmark 60/40
  violaba el límite de 20% por posición. El benchmark es una política de referencia
  y no está sujeto al mandato de la estrategia, así que se exime explícitamente.
  La corrida fallida quedó registrada como `failed`, como se esperaba.
- Smoke test sobre datos sintéticos: corre de extremo a extremo y de forma
  determinista. Sus métricas **no tienen contenido económico**; no se interpretan.

Trials de investigación registrados: **0**.
