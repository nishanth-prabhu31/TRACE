"""Small helpers shared by pipeline steps."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from trace_aml.config import SETTINGS_FILE
from trace_aml.contract import CONTRACT_VERSION


def now_utc() -> str:
    """Current time as an ISO 8601 UTC string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_info(path: Path) -> dict:
    """Name, size and modified time of an input file, for manifests."""
    stat = path.stat()
    modified = datetime.fromtimestamp(stat.st_mtime, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {"file": str(path.name), "bytes": stat.st_size, "modified": modified}


def settings_hash() -> str:
    """Short fingerprint of settings.yaml, so a manifest shows which settings were used."""
    return hashlib.sha256(SETTINGS_FILE.read_bytes()).hexdigest()[:12]


def write_manifest(folder: Path, step: str, started: str, **details) -> Path:
    """Write manifest.json describing what a pipeline step did (docs/03-ARCHITECTURE.md 2.2)."""
    manifest = {
        "step": step,
        "contract_version": CONTRACT_VERSION,
        "settings_hash": settings_hash(),
        "started_at": started,
        "finished_at": now_utc(),
        **details,
    }
    out = folder / f"manifest_{step}.json"
    out.write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return out
