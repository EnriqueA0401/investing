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

## 2026-10-21 — Fase 2: datos reales

- Adaptador `yfinance` (precios, dividendos, volumen) + FRED `DTB3` (T-bill 3M) como cash.
- **Total return reconstruido** a partir del cierre ajustado por splits más las
  distribuciones en su ex-date. Es point-in-time. El "Adj Close" del vendor solo
  se usa como verificación cruzada:
  - Divergencia anualizada ≤ 1 pb/año en los 18 ETFs.
  - Divergencia máxima diaria 0.12% (VGK).
  - La descarga falla si se supera 0.5% diario.
- Splits verificados: EEM y EFA (3:1, 2005), IWM (2:1, 2005), EEM (3:1, 2008),
  EWJ (1:4, 2016) y BIL (1:2, 2017). Sin huecos internos ni precios inválidos.
- Cash: el T-bill observado la sesión anterior se acumula cada día.
- Snapshot congelado `32bc5db46626acc8` (2002-07-01 → 2026-09-30, 6102 sesiones),
  commiteado y fijado en `configs/base.yaml`. El `holdout_end` quedó en
  2026-09-30, el último mes completo en el momento de la congelación.
- El informe de calidad (`docs/data_quality.md`) muestra estadísticas de retorno
  **solo de la ventana research**. Del holdout solo hay chequeos estructurales.
- **Transparencia:** las estadísticas por activo de la ventana research (CAGR,
  vol, drawdown) se vieron antes de pre-registrar las hipótesis de la Fase 3.
  Esas hipótesis vienen de la literatura, no de estos números, pero el comité debe
  saber que el investigador no estaba "ciego" al comportamiento agregado de los
  activos en la muestra research.
- Survivorship: los 18 ETFs siguen cotizando. El universo se fijó ex-ante, así que
  no hay sesgo de selección por supervivencia *dentro* del universo. Sí lo hay en
  la elección de "ETFs conocidos hoy", y queda documentado como limitación.
