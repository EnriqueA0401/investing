# Plan del estudio de acciones long/short (pre-registro)

**Registrado:** 2026-10-24, antes de ejecutar cualquier backtest sobre acciones. El ledger no
contiene ningún trial de acciones en este commit.

## 1. Pregunta
¿Existe un factor o una combinación de factores con tesis publicada que genere, en una cartera
long/short neutral a mercado sobre acciones grandes de EE. UU., un retorno en exceso del cash
estadísticamente defendible tras deflactar por el número de pruebas y neto de costes, préstamo de
títulos y sesgo de supervivencia?

## 2. Datos
- **Precios:** Yahoo Finance. **Fundamentales:** SEC EDGAR XBRL, point-in-time: primer valor
  reportado, usable el día siguiente a la fecha de presentación. **Macro:** FRED BAA10Y y T10Y3M,
  con 1 sesión de retraso. **Cash:** T-bill a 3 meses (DTB3).
- **Universo:** "Russell-1000-like" reconstruido. Las 1000 mayores por capitalización el último día
  de mayo (acciones presentadas a esa fecha × precio), vigente de julio a junio.
- **Filtros de inversión:**
  - Precio ≥ $5.
  - ADV de 63 días ≥ $10M.
  - ≥ 252 días de historia.
  - Se **excluye el sector financiero** (FF12 "Money"): sus ratios contables no son comparables.
- **Sesgo de supervivencia (limitación aceptada por el usuario):** solo hay precios de empresas que
  cotizan hoy. Se mide con `EntityPublicFloat` de todos los filers de EDGAR (cobertura por año en
  `research/results/equity_data/`) y se penaliza según el §5.4.

## 3. Ventanas
- Warm-up desde 2009-06.
- **Research:** decisiones de 2012-01 a 2022-12. Antes no hay XBRL suficiente ni historia de 36
  meses.
- **Holdout:** 2023-01-01 → 2026-09-30, sellado.
- *Advertencia:* el investigador conoce el comportamiento agregado del mercado en 2023–2026 por el
  estudio de ETFs. Los factores a nivel acción no se han observado.

## 4. Candidatos (presupuesto cerrado: 11 configuraciones)

**Construcción común (salvo indicación):**
- Señal neutralizada por sector (FF12) y tamaño, con winsorizado 1/99%.
- `ls_sector_neutral` con 40 largos y 40 cortos, repartidos por sector según su tamaño, equiponderados.
- Beta neutralizada reescalando el lado de mayor beta (beta de 252 días frente a SPY).
- Overlay `vol_target` al 10% (solo reduce exposición).
- **Límites:** gross ≤ 200%, |peso| ≤ 5%, |neto| ≤ 50%.
- Rebalanceo mensual con ejecución t+1.

| ID | Tesis | Señales (signo) |
|---|---|---|
| S-VAL | Value | E/P(+), CFO/P(+), B/P(+) |
| S-QUAL | Calidad | GP/A(+), ROE(+), devengos(−) |
| S-MOM | Momentum | 12-1(+) |
| S-RMOM | Momentum residual | Blitz-Huij-Martens(+) |
| S-LVOL | Bajo riesgo / BAB | beta(−), vol 252d(−) |
| S-INV | Inversión | crecimiento de activos(−) |
| S-COMP | Multifactor | los 5 temas con igual peso (cada tema reparte su peso entre sus señales: value 3, calidad 3, momentum 2 [12-1, residual], riesgo 2 [beta, vol], inversión 1 = 11 señales) |
| S-COMP-OPT | Multifactor optimizado | S-COMP + `ls_optimizer` (modelo PCA de 15 factores, IC 0.03, vol 10%, neutral a dólar, beta y sector) |
| S-COMP-MACRO | Multifactor + macro | S-COMP + `credit_regime` (exposición ×0.5 si el z-score de 3 años de BAA10Y > 1) |
| S-ML-GBM | ML | gradient boosting walk-forward con las 11 señales neutralizadas; reentrenamiento anual, ≥ 36 meses, hiperparámetros fijos |
| S-ML-RIDGE | ML | ridge (α = 1) con el mismo esquema |

**Costes:**
- Half-spread de 1.5 / 3 / 6 pb según ADV (≥ $500M / ≥ $100M / resto).
- Comisión de 0.5 pb y slippage de 1 pb.
- Impacto √ con k = 0.1.
- **Préstamo de títulos de 50 pb/año** sobre el nocional corto (general collateral; los títulos
  difíciles de prestar no se modelan, así que se hace sensibilidad ×3 y ×6).
- AUM de 10M USD.

## 5. Criterios de aceptación (los cuatro obligatorios)
1. **DSR ≥ 0.95** sobre retornos diarios netos en exceso del cash. N = configuraciones de este plan
   presentes en el ledger; V[SR] = varianza entre ellas.
2. **Estabilidad:** Sharpe > 0 en ≥ 3 de 4 subperiodos (2012–14, 2015–17, 2018–20, 2021–22).
3. **PBO < 0.5** (CSCV, 16 bloques, retornos mensuales de las 11 configuraciones).
4. **Supervivencia:** Sharpe > 0 tras restar el arrastre pesimista. Para el año de universo *y*:
   - `drag_y = 2 × 0.30 × (1 − cobertura_y) / (2026.75 − y)`, anual, aplicado a diario.
   - Supuesto: los nombres que faltan desaparecen a ritmo uniforme hasta hoy. Si estaban en cartera,
     el evento les cuesta un 30%: quiebra si estaban en largos, prima de adquisición si estaban en
     cortos.

**Clasificación adicional (no decide la selección):**
- "**Alfa**" si el alfa frente a Fama-French 5 + UMD tiene t ≥ 2 (Newey-West, retornos diarios).
- Si no, "**cosecha de primas conocidas**".

## 6. Regla de selección y holdout
- Entre los validados, el de mayor DSR. Si hay empate (±0.01), el más simple.
- Si ninguno se valida, se evalúa el de mayor DSR con la etiqueta "no validado".
- El holdout se abre una vez, para una sola configuración. En la misma corrida se calculan el
  benchmark (SPY) y la referencia equal-weight long-only del universo.
- **ML:** además del backtest, se compara el IC con la combinación simple mediante CPCV y
  walk-forward estricto, como en H-006.

## 7. Lo que no se hará
- Ajustar parámetros fuera de estas 11 configuraciones.
- Reabrir el holdout.
- Omitir resultados negativos.
