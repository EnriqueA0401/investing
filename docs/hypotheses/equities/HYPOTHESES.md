# Hipótesis del estudio de acciones (pre-registradas 2026-10-24)

Cada tesis incluye su mecanismo, el signo esperado y cómo se refuta (criterios comunes en
`STUDY_PLAN.md` §5). Todas se evalúan como carteras long/short neutrales a beta y a sector.

## S-VAL: Value
- **Tesis:** las acciones baratas por beneficio, caja y valor contable rinden más que las caras.
- **Mecanismo:**
  - Prima por riesgo de distress (Fama-French 1993).
  - Sobre-extrapolación del mal desempeño pasado (Lakonishok, Shleifer y Vishny 1994).
- **Signo esperado:** +.
- **Riesgo conocido:** "década perdida" del value en 2010–2020 (Arnott et al. 2021). Podría no
  aparecer en esta muestra.

## S-QUAL: Calidad / rentabilidad
- **Tesis:** las empresas más rentables por activo (GP/A), con ROE alto y pocos devengos rinden más.
- **Mecanismo:** el mercado infravalora la persistencia de la rentabilidad (Novy-Marx 2013) y se
  fija en el beneficio contable en lugar de en la caja (Sloan 1996).
- **Signo esperado:** + (devengos −).

## S-MOM / S-RMOM: Momentum y momentum residual
- **Tesis:** los ganadores relativos de 12-1 meses siguen ganando el mes siguiente. La versión
  residual elimina la exposición al mercado.
- **Mecanismo:** infrarreacción a noticias de la empresa (Jegadeesh y Titman 1993). El residual
  reduce los crashes (Blitz, Huij y Martens 2011).
- **Signo esperado:** +.
- **Riesgo conocido:** crashes en rebotes (2009, noviembre de 2020).

## S-LVOL: Bajo riesgo / betting-against-beta
- **Tesis:** una cartera larga en baja beta y corta en alta beta, neutral a beta, tiene Sharpe
  positivo.
- **Mecanismo:** restricciones de apalancamiento (Frazzini y Pedersen 2014) y preferencia por
  loterías.
- **Signo esperado:** beta −, vol −.
- **Riesgo conocido:** sensibilidad a tipos y crowding.

## S-INV: Inversión
- **Tesis:** las empresas que más crecen en activos rinden menos.
- **Mecanismo:** según la q-theory, la inversión alta señala un coste de capital bajo (Cooper,
  Gulen y Schill 2008; factor CMA).
- **Signo esperado:** −.

## S-COMP / S-COMP-OPT / S-COMP-MACRO: Combinación
- **Tesis:** los 5 temas tienen baja correlación entre sí, así que combinarlos con igual peso da más
  Sharpe que cada uno por separado (diversificación de primas).
- **OPT:** la optimización con un modelo de riesgo mejora la eficiencia sin cambiar la señal.
- **MACRO:**
  - **Tesis:** recortar la exposición cuando el spread de crédito está estresado reduce las colas
    más de lo que cuesta en retorno.
  - **Refutación adicional:** si no mejora el max drawdown frente a S-COMP, no aporta.

## S-ML-GBM / S-ML-RIDGE: Aprendizaje automático
- **Tesis (escéptica, como H-006):** un modelo entrenado walk-forward con las mismas 11 señales
  **no** supera a su combinación equiponderada.
- **Refutación:** el ML supera a S-COMP en el backtest y además en IC fuera de muestra en CPCV
  **y** en walk-forward estricto.
