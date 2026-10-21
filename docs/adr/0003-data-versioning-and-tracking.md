# ADR 0003 — Versionado de datos y tracking ligeros

**Estado:** aceptado (Fase 1)

## Contexto
El prompt sugiere DVC y MLflow. Con 18 ETFs diarios, cada snapshot pesa ~1–2 MB
en parquet+zstd.

## Decisión
- **Datos:** snapshots inmutables direccionados por contenido
  (`data/snapshots/<hash>/`). Cada panel lleva su checksum SHA-256 en
  `manifest.json` junto con la procedencia, y al cargar se verifica todo. Los
  snapshots se commitean en git, así un tercero reproduce sin depender de que
  yfinance devuelva lo mismo mañana (no lo hará: revisa ajustes).
- **Tracking:** directorio por corrida (`runs/<run_id>/`) con la config resuelta,
  métricas, fingerprint del entorno (versiones y hash de `uv.lock`), estado de git
  y artefactos en parquet. El resumen va al ledger (ADR 0002).
- **Entorno:** `uv.lock` fija todas las versiones; CI usa `uv sync --frozen`.

## Alternativas descartadas (por ahora)
- **DVC:** añade un remoto y comandos extra sin beneficio a este tamaño. Se
  adoptará si los snapshots superan ~50 MB (p. ej. un universo de acciones).
  El formato actual migra trivialmente.
- **MLflow:** servidor/UI pesados para lo que hoy es un JSON por corrida. La
  interfaz `Run.log_metrics/log_frame` está aislada para poder poner MLflow detrás.
