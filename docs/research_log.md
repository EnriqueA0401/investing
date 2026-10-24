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

---

# Estudio 2: acciones long/short (iniciado 2026-10-24)

## Fase 0: alcance (decisiones del usuario)
- Datos gratuitos con mitigación del sesgo de supervivencia.
- Universo tipo Russell 1000.
- Long/short neutral a mercado.
- Rebalanceo mensual con 30–50 nombres por lado.

**Avisos dados al usuario antes de empezar:**
- yfinance no tiene empresas deslistadas. Se comprobó: TWTR, ATVI, XLNX, CELG, SIVB y BBBY no
  tienen precio.
- No hay membresía histórica del Russell gratuita: iShares bloquea sus holdings históricos.
- "Los mejores resultados" no puede ser el objetivo. El objetivo es el mejor resultado *verificado*.

## Fase 1: datos
- Volcado completo de EDGAR companyfacts (1.4 GB, 20 501 entidades). Se extrajeron 10.5 M hechos de
  15 252 empresas, de 2009 a 2026, con su fecha de presentación.
- **Incidencia:** la primera extracción acumulaba todo en memoria (~14 GB proyectados sobre 15 GB
  disponibles). Se detuvo y se reescribió para volcar a parquet por bloques.
- **Fundamentales PIT:**
  - Primer valor reportado por periodo (se ignoran las reexpresiones).
  - Usable el día siguiente a la presentación.
  - TTM = FY + YTD − YTD del año previo.
  - Datos con más de 15 meses de antigüedad se descartan.
- **Capitalización:** acciones de la portada (dei) convertidas a las unidades de split de hoy ×
  precio ajustado por splits.
- **Universo:** el top 1000 por capitalización el último día de mayo, vigente de julio a junio.
- **Candidatos:** empresas que cotizan hoy en NYSE, Nasdaq o CBOE con public float ≥ $500M en algún
  momento. Son 2 691; hay precio para 2 681. Pasaron por el universo 1 686 nombres distintos.
- **Sectores:** SIC de EDGAR mapeado a Fama-French 12 (tabla Siccodes12 verificada).

## Fase 2: pre-registro
Commit `1ccfcd2`, con 0 trials de acciones en el ledger:
- 11 configuraciones.
- Criterios: DSR, estabilidad, PBO y un **arrastre de supervivencia definido de antemano**.
- Clasificación alfa / prima conocida (FF5 + UMD).

Notas de ingeniería:
- Al añadir campos de configuración se protegieron los hashes del estudio de ETFs ya registrados
  (test de regresión).
- Los factores con test de causalidad se precomputan una sola vez. Es equivalente por construcción
  y evita O(T²).

### Data defects found before any stock backtest
These came up when sanity-checking the first snapshot (`821e3d1c…`, discarded without being
used):
1. **XOM, GOOGL, META and WBA were missing from the data.**
   - **GOOGL/META:** companies with several share classes report cover-page shares per class. Those
     dimensional facts are absent from companyfacts, so they had no share count and therefore no
     market cap. *Fix:* chain cover-page → balance-sheet → weighted-average basic shares (new
     extracted tag).
   - **XOM** (and Disney, Medtronic): reorganizations moved the listed company onto a **new CIK**
     with no history. *Fix:* link predecessor CIKs by name similarity (≥ 0.85, ignoring legal
     suffixes) and by the timing between the last old filing and the first new one. Google → Alphabet
     is **not** caught (the name changed), so GOOGL enters the universe late; documented limitation.
   - **WBA:** taken private in 2025. Real survivorship, which is expected.
2. **Survivorship coverage** (the share of the true top 1000 by float that we can price): 60% in
   2012, 68% in 2016, 81% in 2020, 86% in 2022.
   - Part of the "missing" are reorganizations and subsidiaries that file their own 10-K, so it
     overstates the bias. The pre-registered penalty is therefore conservative.
   - The float-weighted coverage is unusable because of unit errors in `EntityPublicFloat` (Viacom
     shows "$12 trillion"). Only the count measure is used, as planned.
3. **Latent bug in the price cache:** it was indexed by chunk position and would have silently
   dropped tickers if the candidate list changed. It is now keyed by content.
4. **Absurd market caps** (STX $17,659T, HBAN $9T): XBRL scale errors in share counts. *Fix:*
   discard values more than 10× away from the company's median, after converting to today's split
   units so real splits don't look like errors.
5. **Share counts with gaps** (BRK-B, F and NKE only 24–60 months; Visa missing). *Fix:* combine
   sources **date by date** (cover page > balance sheet > weighted average > float/price) instead of
   choosing one source per company. Visa now has full history; its float-implied market cap
   understates the real one, but that only affects its rank.
6. Still missing: Google before its 2015/16 reorganization (name change, not linked), UA, FOXA and
   NWSA (shares reported only per class).

**Final snapshot:** `7bf3f31948ed6974` (123 MB).
- 1 721 names passed through the universe.
- Coverage of the true top 1000: 61% in 2012, 66% in 2015, 75% in 2018, 84% in 2021, 86% in 2022.
- *Side effect of survivorship:* our universe goes deeper than the real Russell 1000. The 1000th
  name is worth $0.66B in 2012, against ~$1.5–2B in reality, because survivors fill the gaps.

## 2026-10-24: Equity study run, implementation clarification
- In the first run, S-VAL, S-QUAL, S-MOM and S-RMOM completed. **S-LVOL failed** with a mandate
  violation: net exposure was 51%, against a 50% cap. It is recorded as `failed` in the ledger and does
  not count as a trial.
- **Cause:** the pre-registered spec asked for both beta neutralization (shrinking the higher-beta side)
  and |net| ≤ 50%. For low-beta vs high-beta they conflict (long beta ~0.6 vs short beta ~1.5).
- **Clarification:** the mandate limit prevails. If the net exposure breaches the cap, the larger side
  is shrunk until it fits, accepting a residual beta. This only acts where the mandate would otherwise
  be breached. The configurations that had already completed never breached it, so their results and
  hashes are identical. No config is added or changed.
- The study is re-run in full. The 4 completed configurations reproduce with the same hash, so the
  trial count does not grow.

## 2026-10-24: Equity study, validation (research window 2012–2022)
11 configurations, all of them `ok` in the ledger.

| Config | Sharpe | DSR | Sharpe after survivorship drag | Subperiods > 0 | FF5+UMD alpha (t) |
|---|---|---|---|---|---|
| S-LVOL | 0.60 | 0.70 | 0.40 | 3/4 | +0.8% (0.39) |
| S-MOM | 0.37 | 0.41 | 0.22 | 3/4 | +1.8% (0.80) |
| S-COMP | 0.27 | 0.29 | 0.06 | 3/4 | −2.3% (−1.20) |
| S-COMP-MACRO | 0.26 | 0.28 | 0.04 | 2/4 | −1.8% (−1.04) |
| S-ML-GBM | 0.20 | 0.21 | −0.04 | 2/4 | +1.7% (0.85) |
| S-RMOM | 0.12 | 0.15 | −0.08 | 2/4 | −1.4% (−0.71) |
| S-QUAL | 0.00 | 0.08 | −0.27 | 2/4 | −1.0% (−0.62) |
| S-VAL | −0.03 | 0.06 | −0.23 | 2/4 | −1.5% (−0.71) |
| S-COMP-OPT | −0.11 | 0.04 | −0.25 | 1/4 | −2.7% (−0.79) |
| S-ML-RIDGE | −0.14 | 0.03 | −0.34 | 2/4 | −1.0% (−0.44) |
| S-INV | −0.36 | 0.00 | −0.60 | 2/4 | −3.2% (−1.84) |

- DSR hurdle: Sharpe of 0.44 after 11 trials. **PBO = 0.19.**
- **Verdict: no configuration validated.** No DSR reaches 0.95 and no alpha reaches t ≥ 2.
- **Refutation check:** the loadings point the way the literature says (S-MOM on UMD +0.48, S-INV on CMA
  +0.29, S-QUAL on RMW +0.19, S-VAL on HML +0.17) and market beta is close to 0. The construction is
  correct; the weak result is genuine.
- **Context:** value and investment had their worst decade in large caps over these years. The 2020
  "quant winter" hit S-COMP at −19% and S-COMP-OPT at −24%.
- Costs: 0.7–1.6% a year, with one-way turnover of 3–8× a year. They are not the main cause, since
  gross returns are also small.
- **Pre-registered rule:** S-LVOL (highest DSR, "not validated") is evaluated once in the holdout.
- Before that comes robustness on the research window. Its grid is fixed in `equity_variants`:
  - Costs ×3 and borrow fees ×3/×6.
  - 5-day lag.
  - Quarterly rebalancing.
  - AUM of 1e8/1e9.
  - No vol target.
  - 25/60 names per side.
- The ML IC diagnostic (CPCV vs walk-forward) also runs then.

### 2026-10-24: Equity robustness, ML diagnostic and holdout (S-LVOL)

- **Robustness (research window, post-selection):** S-LVOL Sharpe holds at 0.49–0.66 across the
  fixed grid (costs ×3 0.49, borrow ×3/×6 0.56/0.49, lag 5d 0.51, quarterly 0.62, AUM 1e8/1e9
  0.58/0.52, no vol target 0.55, 25/60 per side 0.66/0.58). Not a parameter artifact within research.
- **ML IC diagnostic:** CPCV ridge IC −0.006 vs composite 0.012 (ridge wins 0/5 paths); strict
  walk-forward 0.006 vs 0.010 (t −0.3); coefficient sign flips up to 14 per feature. The skeptical
  thesis is **not refuted**: no learned model is adopted.
- **Holdout (opened once, ledgered unlock; contamination warning expected, as declared in the plan):**
  S-LVOL CAGR 0.6%, Sharpe −0.41, max DD −18.5%, beta −0.10 vs SPY CAGR 20.0%, Sharpe 1.01. PSR 0.22;
  Sharpe after drag −0.63. Positive in 6 of 7 SPY drops > 5% (Feb–Apr 2025: +1.9% vs −18.8%).
- **Conclusion:** no validated edge in the equity study. S-LVOL behaved as a crash hedge, not as a
  return source, and earned less than cash out of sample. Committee report:
  `docs/report/EQUITY_REPORT.md`.
