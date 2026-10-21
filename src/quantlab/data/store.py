"""Content-addressed, immutable data snapshots.

A snapshot is a directory ``<root>/<hash>/`` with one parquet file per panel
plus ``manifest.json`` holding per-file SHA-256 checksums and provenance. The
snapshot id is a hash over the panel checksums, so identical data always
yields the same id and any tampering is detected on load.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from quantlab.data.market_data import PANEL_FIELDS, MarketData


class SnapshotIntegrityError(RuntimeError):
    pass


def _parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, engine="pyarrow", compression="zstd", index=True)
    return buf.getvalue()


def _frame_digest(df: pd.DataFrame) -> str:
    """Digest of the *values*, independent of parquet writer metadata."""
    h = hashlib.sha256()
    h.update(json.dumps([str(c) for c in df.columns]).encode())
    h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
    return h.hexdigest()


class SnapshotStore:
    def __init__(self, root: Path) -> None:
        self.root = root

    def save(self, data: MarketData, provenance: Mapping[str, Any]) -> str:
        panels = {name: getattr(data, name) for name in PANEL_FIELDS}
        digests = {name: _frame_digest(df) for name, df in panels.items()}
        asset_class = dict(sorted(data.asset_class.items()))
        snap_id = hashlib.sha256(
            json.dumps({"panels": digests, "asset_class": asset_class}, sort_keys=True).encode()
        ).hexdigest()[:16]
        target = self.root / snap_id
        if (target / "manifest.json").exists():
            self.load(snap_id)  # verifies integrity of the existing copy
            return snap_id
        target.mkdir(parents=True, exist_ok=True)
        for name, df in panels.items():
            (target / f"{name}.parquet").write_bytes(_parquet_bytes(df))
        manifest = {
            "snapshot_id": snap_id,
            "created_at": datetime.now(UTC).isoformat(),
            "panels": digests,
            "asset_class": asset_class,
            "start": str(data.dates[0].date()),
            "end": str(data.end.date()),
            "tickers": data.tickers,
            "provenance": dict(provenance),
        }
        (target / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        return snap_id

    def manifest(self, snap_id: str) -> dict[str, Any]:
        path = self.root / snap_id / "manifest.json"
        if not path.exists():
            raise FileNotFoundError(f"snapshot {snap_id} not found under {self.root}")
        manifest: dict[str, Any] = json.loads(path.read_text())
        return manifest

    def load(self, snap_id: str) -> MarketData:
        manifest = self.manifest(snap_id)
        panels: dict[str, pd.DataFrame] = {}
        for name in PANEL_FIELDS:
            df = pd.read_parquet(self.root / snap_id / f"{name}.parquet")
            if _frame_digest(df) != manifest["panels"][name]:
                raise SnapshotIntegrityError(
                    f"snapshot {snap_id}: panel '{name}' checksum mismatch"
                )
            panels[name] = df
        return MarketData(**panels, asset_class=manifest["asset_class"])
