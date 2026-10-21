# Plan del estudio (pre-registro)

**Registrado:** 2026-10-21, antes de ejecutar cualquier backtest de estrategia sobre datos reales
(el ledger `research/trials.jsonl` está vacío en este commit).

## 1. Pregunta
¿Existe alguna regla sistemática, basada en tesis publicadas, que mejore **de forma
estadísticamente defendible y neta de costes** a una cartera ingenua equal-weight
(y al benchmark 60/40), dentro del mandato long-only, sin apalancamiento y con un
máximo de 20% por posición, sobre 17 ETFs multi-activo con rebalanceo mensual?

## 2. Datos y ventanas
- Snapshot `32bc5db46626acc8`.
- Warm-up desde 2002-07-01. Las decisiones evaluables empiezan en 2005-01.
- **Research:** hasta 2022-12-31. **Holdout sellado:** 2023-01-01 → 2026-09-30.
- **Ajuste del universo:** BIL deja de ser invertible como activo de riesgo
  (`tradable: false`). Motivo: desde la Fase 2 el cash está modelado
  explícitamente con el T-bill, y mantener BIL (vol ~0.5%) haría que cualquier
  ponderación por riesgo convergiera trivialmente a cash. Decisión tomada antes de
  cualquier backtest. Universo de riesgo: 17 ETFs.

## 3. Candidatos (presupuesto cerrado: 15 configuraciones)

| ID config | Hipótesis | Señal / overlay | Construcción |
|---|---|---|---|
| `B-EW` | Baseline | ninguna | 1/N |
| `H1-TS6`, `H1-TS12` | H-001 trend | filtro TSMOM 6m / 12m (exceso sobre cash) | 1/N, slot a cash si el filtro está apagado |
| `H2-XS12`, `H2-XS6` | H-002 momentum XS | top-6 por momentum 12-1 / 6-1 | 1/N entre los seleccionados |
| `H2-XS12-RMV` | H-002 | momentum 12-1 como alfa (Grinold, IC=0.05) | media-varianza robusta (cov. Ledoit-Wolf) |
| `H3-IV`, `H3-ERC`, `H3-HRP`, `H3-MV` | H-003 riesgo | ninguna | inverse-vol / ERC / HRP / mínima varianza (cov. LW) |
| `H4-IV-TS12`, `H4-ERC-TS12` | H-004 combinación | filtro TSMOM 12m | inverse-vol / ERC |
| `H5-EW-VT` | H-005 vol targeting | escala a vol objetivo 10% (máx. 1×) | 1/N |
| `H6-COMBO`, `H6-RIDGE` | H-006 ML | top-6 por combinación z-score igual / ridge walk-forward purgado | 1/N entre los seleccionados |

Parámetros comunes (fijos):
- Covarianza Ledoit-Wolf sobre 252 días.
- Vol para inverse-vol: 63 días.
- Rebalanceo mensual con ejecución t+1.
- Costes de `configs/base.yaml`.
- AUM de 1M USD.

Cualquier configuración adicional se registra primero aquí y cuenta en el ledger.

## 4. Métricas y tests (fijados ahora)
- **Métrica primaria:** Sharpe anualizado de los retornos diarios netos en exceso
  sobre el T-bill, en la ventana research.
- **Criterios de aceptación** (los cuatro obligatorios):
  1. **Deflated Sharpe Ratio ≥ 0.95**, con N = nº de configuraciones research en
     el ledger al momento de la selección y la varianza de los Sharpe entre trials.
  2. **Mejora sobre B-EW:** diferencia de Sharpe > 0 con p < 0.05 (bootstrap
     circular por bloques de retornos mensuales, bloque de 6 meses, 10 000
     remuestreos, unilateral).
  3. **Estabilidad:** Sharpe en exceso > 0 en al menos 3 de 4 subperiodos
     (2005–09, 2010–14, 2015–19, 2020–22).
  4. **Robustez del proceso de selección:** PBO (CSCV, 16 bloques) del conjunto
     de candidatos < 0.5. Si PBO ≥ 0.5, ningún candidato se declara validado.
- **Complementarios** (no deciden):
  - Sortino, Calmar, max drawdown, turnover y coste.
  - Selección walk-forward anual: elegir cada año el mejor por Sharpe trailing y
    medir el año siguiente.
  - Test contra 1000 carteras aleatorias con las mismas restricciones.
  - Capacidad y sensibilidad a costes ×2 y ×3.

## 5. Regla de selección para el holdout (fijada ahora)
- Si ≥ 1 candidato cumple los 4 criterios, se elige el de **mayor DSR**. Si hay
  empate (±0.01), el más simple (menos parámetros).
- Si ninguno cumple, se evalúa en el holdout el candidato de mayor DSR, etiquetado
  **"no validado"**, para medir la degradación. La recomendación por defecto pasa
  a ser la cartera baseline.
- El holdout se abre **una sola vez y para una sola configuración**. B-EW y el 60/40
  se calculan dentro de esa misma corrida como referencias, sin un desbloqueo extra.
- El análisis de robustez (Fase 6) se hace sobre research **después** de la
  selección. No puede cambiar la configuración seleccionada; solo informa de los
  riesgos.

## 6. Lo que NO se hará
- No se ajustarán parámetros mirando resultados de research más allá de las 15
  configuraciones registradas.
- No se re-abrirá el holdout tras ver su resultado.
- No se omitirán los resultados de los candidatos fallidos.
