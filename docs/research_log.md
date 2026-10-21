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

## 2026-10-21 — Fase 3: pre-registro

- Commit `41df1f3`: se registraron H-001…H-006 y `STUDY_PLAN.md` (15 configuraciones,
  criterios de aceptación y regla de selección para el holdout) con el ledger vacío.
- BIL pasa a `tradable: false`. El cash ya se modela con el T-bill; mantener BIL
  haría que la ponderación por riesgo convergiera trivialmente a cash. La decisión
  es anterior a cualquier backtest.
- El código de factores, constructores, overlays, señal ML y estadística (DSR, PBO,
  CPCV) se commiteó (`5e73520`) antes de correr el estudio.

## 2026-10-21 — Fase 5: estudio y validación (ventana research 2005–2022)

Se corrieron las 15 configuraciones registradas. Trials en el ledger: **15**.

Detalles fijados antes de la primera evaluación, donde el plan no los precisaba:
- PBO sobre retornos **mensuales** en exceso.
- Bootstrap con p-valor centrado (Ledoit-Wolf 2008).

| Config | Sharpe | DSR | ΔSR vs EW | p | Subperiodos > 0 |
|---|---|---|---|---|---|
| H4-IV-TS12 | 0.66 | 0.979 | +0.25 | 0.095 | 4/4 |
| H4-ERC-TS12 | 0.62 | 0.972 | +0.20 | 0.163 | 4/4 |
| H1-TS6 | 0.60 | 0.966 | +0.27 | 0.108 | 4/4 |
| H1-TS12 | 0.55 | 0.948 | +0.20 | 0.143 | 4/4 |
| H6-COMBO | 0.50 | 0.919 | +0.13 | 0.265 | 3/4 |
| H3-IV | 0.48 | 0.908 | 0.00 | 0.514 | 3/4 |
| H2-XS12-RMV | 0.47 | 0.903 | −0.01 | 0.540 | 3/4 |
| H2-XS12 | 0.47 | 0.903 | +0.08 | 0.315 | 3/4 |
| H3-ERC | 0.47 | 0.901 | −0.02 | 0.563 | 3/4 |
| H2-XS6 | 0.45 | 0.885 | +0.06 | 0.360 | 3/4 |
| **B-EW** | 0.43 | 0.867 | — | — | 4/4 |
| H5-EW-VT | 0.42 | 0.862 | +0.01 | 0.430 | 3/4 |
| H3-HRP | 0.42 | 0.853 | −0.07 | 0.647 | 3/4 |
| H3-MV | 0.38 | 0.816 | −0.11 | 0.750 | 3/4 |
| H6-RIDGE | 0.32 | 0.739 | −0.10 | 0.641 | 4/4 |

Referencia: el 60/40 obtuvo Sharpe 0.52 en la misma ventana.

**PBO = 0.60** (12 870 combinaciones CSCV):
- Probabilidad de pérdida fuera de muestra del mejor in-sample: 2.5%.
- Mediana del Sharpe fuera de muestra del mejor in-sample: 0.55.

**Veredicto: ningún candidato queda validado.**
- Ninguno supera a EW con p < 0.05.
- PBO ≥ 0.5: ordenar candidatos por su backtest no es mejor que el azar. Todos
  son carteras multi-activo long-only con betas similares; la diferencia entre
  ellos es ruido frente a lo que tienen en común.
- DSR ≥ 0.95 en H4-IV-TS12, H4-ERC-TS12 y H1-TS6. Su exceso sobre el cash es
  significativo tras deflactar por 15 trials, pero eso **no** demuestra que
  mejoren a EW.

Selección walk-forward ("cada año, el ganador del backtest hasta la fecha"):
- Sharpe 0.42, frente a 0.47 de la mezcla equiponderada de todos los candidatos
  y 0.37 de EW en la misma ventana.
- Elegir al ganador no añade valor; diversificar entre reglas sí lo hace.

Regla pre-registrada:
- Se evalúa en el holdout **H4-IV-TS12** (mayor DSR), con la etiqueta **"no validado"**.
- La recomendación por defecto pasa a ser la cartera baseline (B-EW).

H-006 (ML): ridge es el peor candidato (Sharpe 0.32, turnover 4×/año). La tesis
escéptica se sostiene en cartera; el diagnóstico CPCV se reporta en la Fase 6.

Grid de robustez (fijado en `quantlab/robustness.py::standard_variants`, commiteado
antes de correrlo). Para H4-IV-TS12:
- Lookback de tendencia {126, 189, 252, 315} × ventana de vol {21, 63, 126}.
- Costes ×2, ×3 y ×5.
- Lag de 2 y 5 días.
- Rebalanceo trimestral.
- AUM de 1e7 a 1e10.
- Tope de turnover del 10%.

Para B-EW: costes, lag, trimestral, AUM y tope de turnover.

Todas estas variantes quedan en el ledger. No son candidatas: el DSR de la
selección usa N = 15. Como sensibilidad, también se reporta con N = total del ledger.

## 2026-10-21 — Fase 6: robustez (ventana research, antes de abrir el holdout)

H4-IV-TS12, candidato seleccionado pero no validado:
- **Parámetros:** Sharpe entre 0.56 y 0.72 en las 12 combinaciones de lookback de
  tendencia × ventana de vol. Todas superan a EW (0.43). La original (252/63)
  queda en la mitad del rango; no es un pico aislado.
- **Costes:** ×2 → 0.64, ×3 → 0.63, ×5 → 0.60.
- **Ejecución:** lag de 2 días → 0.66; de 5 días → 0.63.
- **Rebalanceo trimestral:** 0.50, con drawdown de −14%. Es el punto más débil.
- **Capacidad:** con 1e10 USD el Sharpe es 0.60 y el coste 0.24%/año. No hay
  límite de capacidad relevante para el mandato.
- **Tope de turnover del 10%:** 0.68, con turnover anual 0.74.

Test contra filtros aleatorios con el mismo número de activos apagados por mes
(2000 simulaciones, mensual, bruto):
- Sharpe observado 0.72, frente a una media aleatoria de 0.57 (p95 0.67).
  **p = 0.0075**: la elección del filtro de tendencia aporta información.
- Sesgo: el test se hizo sobre el candidato seleccionado por mayor DSR, así que
  es optimista.

Exposiciones (OLS mensual con Newey-West sobre los factores equity, duration,
credit, commodity y gold):
- H4-IV-TS12: **alfa +0.71%/año, t = 1.21 (no significativo)**, R² 0.56. Su
  ventaja se explica sobre todo por una exposición defensiva y dinámica, no por
  alfa independiente.
- B-EW: alfa −1.6%/año (t = −3.3) frente a esos mismos factores.
- 60/40: alfa 0.0.

Stress histórico (research):

| Episodio | H4 | EW | 60/40 |
|---|---|---|---|
| GFC | +5.3% | −29.8% | −35.1% |
| COVID | −6.9% | −18.3% | −21.2% |
| 2022 | −1.9% | −19.9% | −20.4% |

La protección es real, pero se paga con menor retorno en todos los regímenes:
CAGR 3.5% frente a 5.2% (EW) y 6.8% (60/40).

**H-006, diagnóstico ML:**
- CPCV (6 grupos, k = 2, purga + embargo 21d): IC de ridge 0.046 frente a 0.022
  de la combinación simple, con ridge ganando en 5 de 5 caminos. Por el criterio
  pre-registrado, la tesis escéptica queda **refutada a nivel de IC**.
- El walk-forward con solo pasado (el despliegue real) **no lo confirma**: IC de
  ridge 0.020 frente a 0.017, con t de la diferencia 0.07. Los coeficientes
  cambian de signo entre 3 y 6 veces.
- Conclusión: la ventaja en CPCV viene de entrenar con bloques posteriores al de
  test (información de régimen). No es desplegable, y ML no se adopta.
- Lección metodológica: el CPCV sirve para medir la dispersión del rendimiento,
  pero en series con no-estacionariedad hay que contrastarlo con un walk-forward
  estricto.

Trials en el ledger tras la robustez: 49 (15 de selección + 34 de robustez).

**Siguiente paso:** abrir el holdout una única vez para H4-IV-TS12, con B-EW y el
60/40 calculados en la misma corrida.

## 2026-10-21 — Fase 6b: holdout (evaluado una vez)

- Desbloqueo registrado en el ledger para H4-IV-TS12, con la etiqueta "no validado".
  1 desbloqueo, 1 trial final.
- Holdout 2023-02 → 2026-09 (44 meses):

| Cartera | Sharpe | CAGR | Max DD |
|---|---|---|---|
| H4-IV-TS12 | 0.41 | 6.1% | −3.6% |
| B-EW | 0.54 | 8.9% | −7.7% |
| 60/40 | 0.83 | 12.6% | −11.2% |

- H4 frente a EW: ΔSR −0.11, IC 95% [−0.54, +0.29]. Ninguna diferencia es significativa
  con 44 meses, pero la dirección confirma la predicción del PBO: el ganador del
  backtest no lo es fuera de muestra.
- Pesos actuales de H4-IV-TS12: **80% cash**. El filtro apaga todos los bonos porque su
  retorno de 12 meses está por debajo del T-bill. No es una asignación compatible con un
  objetivo de vol del 10%.

## 2026-10-21 — Fase 7: conclusión

- **No hay edge validado.**
- Recomendación según la regla pre-registrada: **cartera de política equal-weight** sobre
  los 17 ETFs de riesgo (5.88% cada uno, rebalanceo mensual).
- La elección frente al 60/40 es estratégica, no estadística (ver el informe, §9).
- Informe: `docs/report/REPORT.md`.
- Trials totales: 49 de research (15 de selección + 34 de robustez) y 1 final.
- Hallazgos negativos que el comité debe conservar:
  1. Las reglas ganadoras en research ganaban por reducir riesgo, no por alfa (t = 1.2).
  2. El ML pareció funcionar en CPCV y no funciona en walk-forward estricto.
  3. Elegir al ganador del backtest es peor que promediar todas las reglas.
