# H-002 — Momentum cross-sectional entre clases de activo

| Campo | Valor |
|---|---|
| ID | H-002 |
| Fecha de registro | 2026-10-21 |
| Estado | registrada |
| Configs | `H2-XS12`, `H2-XS6`, `H2-XS12-RMV` |

## 1. Tesis
Los ETFs con mejor retorno relativo en los últimos 12 meses (excluyendo el último)
superan a los de peor retorno relativo durante el mes siguiente. Una cartera
concentrada en el tercio superior supera a equal-weight en Sharpe neto.

## 2. Mecanismo
Conductual: difusión lenta de la información entre mercados y capital que se mueve
despacio entre clases de activo. Se excluye el último mes para evitar la reversión
de corto plazo (liquidez / microestructura).

## 3. Predicción
- Signo positivo entre el ranking de momentum 12-1 y el retorno relativo del mes siguiente.
- Riesgo conocido: *momentum crashes* en los rebotes tras grandes caídas (2009).
- Turnover mayor que el de los baselines; el coste debe absorberse.

## 4. Refutación
`STUDY_PLAN.md` §4.

## 5. Especificación
- Señal: `TR(t−21)/TR(t−L) − 1`, con L = 252 (12-1) o 126 (6-1).
- `H2-XS12` y `H2-XS6`: top-6 de los invertibles, 1/6 cada uno (16.7% < 20%).
- `H2-XS12-RMV`: α_i = IC · σ_i · z_i, con IC = 0.05 (Grinold), z = z-score
  cross-sectional, σ = vol diaria de 63d. Maximiza `α'w − (λ/2) w'Σw` con λ = 10,
  Σ Ledoit-Wolf (252d), `Σw = 1`, `0 ≤ w ≤ 20%`.
- Presupuesto: 3 variantes.

## 6. Literatura
- Asness, Moskowitz & Pedersen (2013), *Value and Momentum Everywhere*, JF.
- Daniel & Moskowitz (2016), *Momentum Crashes*, JFE.

El momentum entre clases de activo es más débil y ruidoso que dentro de acciones;
con 17 activos la amplitud es baja (ley fundamental: IR ≈ IC·√breadth).
