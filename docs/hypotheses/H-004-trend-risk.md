# H-004 — Ponderación por riesgo con filtro de tendencia

| Campo | Valor |
|---|---|
| ID | H-004 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H4-IV-TS12`, `H4-ERC-TS12` |

## 1. Tesis
Combinar H-003 (asignación por riesgo) con el filtro de H-001 (12m) mejora el
Sharpe y reduce el drawdown frente a cada componente por separado y frente a 1/N.

## 2. Mecanismo
Diversificación entre una prima estructural (aversión al apalancamiento) y un
efecto conductual (tendencia) con baja correlación. El filtro corta el riesgo de
duración en regímenes de tipos al alza persistentes, la principal debilidad de H-003.

## 3. Predicción
- Sharpe(H4) > max(Sharpe(H1-TS12), Sharpe(H3-*)) en la misma construcción.
- Menor drawdown en 2022.

## 4. Refutación
`STUDY_PLAN.md` §4. Además, si no mejora a ambos componentes, la combinación no
aporta y se prefiere el componente más simple.

## 5. Especificación
- Pesos de H3-IV / H3-ERC. Se aplica el filtro TSMOM 12m de H-001 y los slots
  apagados van a cash (sin redistribuir).
- Presupuesto: 2 variantes.

## 6. Literatura
- Clare, Seaton, Smith & Thomas (2016), *The trend is our friend: Risk parity, momentum and trend following in global asset allocation*.
