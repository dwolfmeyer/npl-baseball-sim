"""Raw snapshot store.

Every pull is written exactly as received to
`<root>/<source>/<dataset>/<season>/<key>.<ext>` and logged in
`<root>/manifest.jsonl` with its URL, fetch time and SHA-256, so any week can
be re-normalized (or re-simmed) from the bytes we actually saw. Frames that
come back from pybaseball are stored as Parquet exactly as returned.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

DEFAULT_RAW_ROOT = Path("data/raw")


class SnapshotStore:
    def __init__(self, root: Path | str = DEFAULT_RAW_ROOT):
        self.root = Path(root)

    def path(self, source: str, dataset: str, season: int, key: str, ext: str) -> Path:
        return self.root / source / dataset / str(season) / f"{key}.{ext}"

    def has(self, source: str, dataset: str, season: int, key: str, ext: str) -> bool:
        return self.path(source, dataset, season, key, ext).exists()

    def put_bytes(
        self, source: str, dataset: str, season: int, key: str, ext: str, data: bytes, url: str
    ) -> Path:
        path = self.path(source, dataset, season, key, ext)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(data)
        tmp.replace(path)
        self._log(source, dataset, season, key, path, data, url)
        return path

    def put_frame(
        self, source: str, dataset: str, season: int, key: str, frame: pd.DataFrame, url: str
    ) -> Path:
        buf = io.BytesIO()
        frame.to_parquet(buf, index=False)
        return self.put_bytes(source, dataset, season, key, "parquet", buf.getvalue(), url)

    def read_json(self, source: str, dataset: str, season: int, key: str):
        with open(self.path(source, dataset, season, key, "json")) as fh:
            return json.load(fh)

    def read_frame(self, source: str, dataset: str, season: int, key: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(source, dataset, season, key, "parquet"))

    def keys(self, source: str, dataset: str, season: int) -> list[str]:
        folder = self.root / source / dataset / str(season)
        if not folder.exists():
            return []
        return sorted(p.stem for p in folder.iterdir() if not p.name.endswith(".tmp"))

    def latest_key(self, source: str, dataset: str, season: int, on_or_before: str) -> str | None:
        """Newest key (ISO date keys sort by date) not after `on_or_before`."""
        keys = [k for k in self.keys(source, dataset, season) if k <= on_or_before]
        return keys[-1] if keys else None

    def _log(self, source, dataset, season, key, path: Path, data: bytes, url: str) -> None:
        entry = {
            "source": source,
            "dataset": dataset,
            "season": season,
            "key": key,
            "path": str(path.relative_to(self.root)),
            "url": url,
            "fetched_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "sha256": hashlib.sha256(data).hexdigest(),
            "bytes": len(data),
        }
        self.root.mkdir(parents=True, exist_ok=True)
        with open(self.root / "manifest.jsonl", "a") as fh:
            fh.write(json.dumps(entry) + "\n")
