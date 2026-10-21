# H-006 — ¿Aporta un modelo aprendido sobre una combinación simple de señales?

| Campo | Valor |
|---|---|
| ID | H-006 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H6-COMBO`, `H6-RIDGE` |

## 1. Tesis (deliberadamente escéptica)
Una regresión ridge entrenada walk-forward sobre cinco features cross-sectional
**no** predice el retorno relativo del mes siguiente mejor que su combinación
equiponderada (sin entrenamiento). Se registra para justificar, o descartar, el
uso de ML en este problema.

## 2. Mecanismo
Ley fundamental de la gestión activa: con 17 activos y ~200 meses, la relación
señal/ruido es muy baja. El error de estimación de los coeficientes suele anular
la ventaja de ponderarlos de forma óptima (análogo a DeMiguel et al. 2009 para pesos).

## 3. Predicción
- IC fuera de muestra de ridge ≤ IC de la combinación simple, en CPCV y en walk-forward.
- El ML solo se adoptaría si mejora la combinación simple de forma clara y estable
  **y además** cumple los criterios comunes.

## 4. Refutación
- La tesis escéptica se refuta si la mediana del IC ridge en CPCV supera a la de
  COMBO **y** ridge supera a COMBO en ≥ 4 de 5 caminos.
- Para su uso en cartera: `STUDY_PLAN.md` §4.

## 5. Especificación
- Features en cada fecha de rebalanceo s, con z-score cross-sectional: momentum
  12-1, momentum 6-1, retorno de 1 mes (reversión), vol de 63d y TSMOM 12m.
- Label: retorno del activo de s al siguiente rebalanceo s', menos la media
  cross-sectional.
- `H6-RIDGE`:
  - Ridge con α = 1.0 (fijo, sin tuning) y sin intercepto.
  - Entrenamiento en `t` solo con muestras cuyo label termina ≤ `t` (purga).
  - Mínimo 36 meses de entrenamiento; antes, la señal es la combinación simple.
- `H6-COMBO`: media de los z-scores con el signo teórico (+mom12, +mom6, −rev1m,
  −vol, +tsmom).
- Ambas: top-6 equiponderado.
- Validación adicional del modelo:
  - CPCV con N = 6 grupos y k = 2 (15 splits, 5 caminos).
  - Purga de solapes de label y embargo de 1 mes.
  - Métrica: IC de Spearman fuera de muestra.
- Presupuesto: 2 variantes de cartera. El CPCV es diagnóstico y no genera configuraciones.

## 6. Literatura
- López de Prado (2018), *Advances in Financial Machine Learning*, cap. 7 y 12 (purged CV, CPCV).
- Grinold & Kahn, *Active Portfolio Management* (IR = IC·√breadth).
