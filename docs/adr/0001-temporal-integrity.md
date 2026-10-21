# ADR 0001 — Integridad temporal por construcción

**Estado:** aceptado (Fase 1)

## Contexto
El look-ahead es el fallo más caro y más silencioso de un backtest. Confiar en la
disciplina de quien escribe cada factor no basta.

## Decisión
1. **Un único punto de acceso a datos.** Solo el orquestador (`pipeline.py`) ve el
   dataset completo. En cada fecha de decisión `t` construye `data.asof(t)` y eso
   es lo único que reciben los componentes. `DecisionContext` lanza error si la
   historia termina después de `as_of`.
2. **Convención de timing explícita.** Señal con información hasta el cierre de
   `t`; ejecución al cierre de `t + execution_lag_days` (mínimo 1, validado).
   El día de la ejecución la cartera gana el retorno de las posiciones *previas*.
3. **Costes con información previa.** ADV y volatilidad usadas en el coste del día
   `d` se miden hasta `d-1`.
4. **Fechas de rebalanceo causales.** Un día es fin de periodo solo si la
   siguiente sesión observada cae en otro periodo; el periodo final incompleto
   nunca se trata como cerrado.
5. **Test adversarial obligatorio.** Cada factor debe pasar `assert_causal`, que
   compara su salida con datos completos, truncados y con el futuro aleatorizado
   en cortes repartidos por toda la muestra.
6. **Precios ajustados.** Las series ajustadas por dividendos están
   retro-ajustadas: sus niveles pasados cambian. Solo se permiten ratios/retornos.

## Consecuencias
- Recalcular factores en cada fecha es O(T²); con ~20 activos y ~250 meses es
  despreciable (segundos). Si el universo crece, se podrá vectorizar *siempre que*
  el factor siga pasando `assert_causal`.
- Datos macro con revisiones (FRED) necesitarán vintages (ALFRED) — Fase 2.
