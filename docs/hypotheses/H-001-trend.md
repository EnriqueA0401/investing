# H-001 — Time-series momentum (filtro de tendencia)

| Campo | Valor |
|---|---|
| ID | H-001 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H1-TS6`, `H1-TS12` |

## 1. Tesis
Un activo cuyo retorno total de los últimos 12 meses (6 en la variante) supera al
cash tiende a seguir superándolo el mes siguiente. Si se sale de los activos con
tendencia negativa, el Sharpe neto supera al de equal-weight, sobre todo por menor
drawdown.

## 2. Mecanismo
Conductual: infrarreacción inicial a la información (anclaje, efecto disposición)
y sobrerreacción tardía (*herding*). Hay además flujos de cobertura y de
rebalanceo de riesgo pro-cíclicos (vol targeting institucional, CTA). No es una
prima de riesgo clásica: paga en crisis prolongadas (convexidad *crisis alpha*) y
pierde en giros bruscos (*whipsaw*).

## 3. Predicción
- Signo: positivo (exceso de retorno trailing > 0 implica mayor retorno esperado).
- Mejora esperada concentrada en 2008 y en mercados bajistas largos.
- Coste: menor retorno en mercados alcistas con correcciones en V (2020).

## 4. Refutación
Criterios comunes de `STUDY_PLAN.md` §4.

## 5. Especificación
- Señal: `TR(t)/TR(t−L) − 1 − (Cash(t)/Cash(t−L) − 1)`, con L = 126 o 252 sesiones.
- Overlay: cada activo recibe su slot 1/N. Si la señal es ≤ 0 en `t`, el slot va a cash.
- Sin skip-month: la literatura de TSMOM no lo usa.
- Presupuesto: 2 variantes (L = 6m, 12m).

## 6. Literatura
- Moskowitz, Ooi & Pedersen (2012), *Time Series Momentum*, JFE.
- Hurst, Ooi & Pedersen (2017), *A Century of Evidence on Trend-Following Investing*.
- Faber (2007), *A Quantitative Approach to Tactical Asset Allocation*.

Hay evidencia de rendimiento débil de los CTA en 2012–2018 (posible decaimiento o crowding).
