# ADR 0002 — Holdout sellado y ledger de pruebas

**Estado:** aceptado (Fase 1)

## Contexto
El Deflated Sharpe Ratio y el PBO necesitan el número real de configuraciones
probadas. Si ese número se puede olvidar o borrar, la corrección pierde sentido.
Además, el holdout solo vale si se mira una vez.

## Decisión
- **Ledger append-only** (`research/trials.jsonl`, versionado en git). Cada corrida
  añade una línea con hash de config, hipótesis, etapa, estado (ok/failed), commit
  y métricas. Un *trial* = un hash de config distinto que terminó OK.
  - Las corridas fallidas se registran (para auditoría), pero no cuentan como trial.
  - El nombre del experimento no entra en el hash: renombrar no "resetea" el contador.
  - Un hook de pre-commit y CI verifican que nunca se reescriba ni se trunque.
- **Holdout sellado** (`HoldoutGuard`):
  - La etapa `research` solo ve datos hasta `research_end`.
  - La etapa `final` exige un desbloqueo explícito con motivo, que queda en el ledger.
  - Re-ejecutar la misma config sobre el holdout es idempotente.
  - Evaluar una config *distinta* se permite (prohibirlo solo invita a saltarse la
    herramienta), pero queda marcado como **contaminación del holdout** y se
    reportará en el informe final.
- Los smoke tests usan un ledger temporal: no son investigación.

## Consecuencias
- El ledger crece sin límite; es texto y pesa poco.
- El conteo es por configuración, no por "idea": variantes de una misma hipótesis
  cuentan todas, que es lo conservador.
