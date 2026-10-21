# H-005 — Volatility targeting (sin apalancamiento)

| Campo | Valor |
|---|---|
| ID | H-005 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H5-EW-VT` |

## 1. Tesis
Reducir la exposición de la cartera 1/N cuando su volatilidad estimada supera el
10% anual (escala = min(1, 10%/σ̂), con el resto en cash) mejora el Sharpe neto y
reduce las colas.

## 2. Mecanismo
La volatilidad es persistente (clustering) y está negativamente correlacionada
con el retorno futuro de los activos de riesgo (efecto apalancamiento), así que
las épocas de alta vol tienen peor ratio retorno/riesgo.

## 3. Predicción
- Menor drawdown y menor curtosis.
- Mejora de Sharpe pequeña: sin apalancamiento solo puede *recortar*, y la
  literatura muestra mejoras más claras en acciones que en multi-activo.

## 4. Refutación
`STUDY_PLAN.md` §4.

## 5. Especificación
- σ̂: vol de la cartera objetivo con Σ Ledoit-Wolf de 63 sesiones (ventana corta
  para reaccionar).
- Exposición máxima de 1×.
- Presupuesto: 1 variante.

## 6. Literatura
- Moreira & Muir (2017), *Volatility-Managed Portfolios*, JF.
- Harvey, Hoyle, Korgaonkar, Rattray, Sargaison & Van Hemert (2018), *The Impact of Volatility Targeting*.
- Cederburg et al. (2020): las mejoras de Moreira-Muir no son robustas fuera de muestra.
