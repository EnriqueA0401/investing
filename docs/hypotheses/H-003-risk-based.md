# H-003 — Ponderación por riesgo (baja volatilidad / risk parity)

| Campo | Valor |
|---|---|
| ID | H-003 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H3-IV`, `H3-ERC`, `H3-HRP`, `H3-MV` |

## 1. Tesis
Ponderar por riesgo (inverse-vol, igual contribución al riesgo, HRP, mínima
varianza) produce un Sharpe neto mayor que ponderar por capital (1/N).

## 2. Mecanismo
Aversión al apalancamiento: los inversores con restricciones de apalancamiento
sobreponderan activos de alto riesgo para alcanzar su retorno objetivo, lo que
abarata (sube el retorno ajustado por riesgo de) los activos de bajo riesgo. Es
una prima ligada a una restricción de financiación.

## 3. Predicción
- Sharpe mayor que 1/N; retorno absoluto menor, porque el mandato no permite
  apalancar hasta la vol objetivo.
- Riesgo conocido: concentración en duración. Pierde en shocks de tipos al alza
  (2013, 2022).

## 4. Refutación
`STUDY_PLAN.md` §4.

**Advertencia ex-ante:** sin apalancamiento este enfoque **no puede** alcanzar el
objetivo de 10% de vol. Aunque gane en Sharpe, puede no cumplir el objetivo de
retorno del mandato. Se reportará de forma explícita.

## 5. Especificación
- Covarianza: Ledoit-Wolf sobre retornos diarios de 252 sesiones; inverse-vol con vol de 63 sesiones.
- Construcciones:
  - Inverse-vol: `w ∝ 1/σ`.
  - ERC: Spinu (2013), con presupuestos de riesgo iguales.
  - HRP: López de Prado (2016), con single linkage y distancia de correlación.
  - Mínima varianza: SLSQP.
- Límite del 20% por posición con redistribución del exceso; si no cabe, el resto va a cash.
- Presupuesto: 4 variantes.

## 6. Literatura
- Frazzini & Pedersen (2014), *Betting Against Beta*.
- Asness, Frazzini & Pedersen (2012), *Leverage Aversion and Risk Parity*.
- López de Prado (2016), *Building Diversified Portfolios that Outperform OOS*.
- DeMiguel, Garlappi & Uppal (2009): 1/N es difícil de batir fuera de muestra.
