"""Run bundles: everything needed to re-create an analysis.

A bundle is deliberately more than "the results".  It holds the spec, the data,
the sampler settings *including the resolved seed*, the exact generated Stan
source, and the toolchain versions that compiled it.  Without the toolchain
versions the word "reproducible" would be false: the same spec compiled by a
different CmdStan is a different program.

Everything is written as plain JSON and Parquet/CSV inside a single directory,
so a bundle stays readable without this package.
"""

from __future__ import annotations

import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .compile import toolchain_fingerprint
from .spec import SCHEMA_VERSION, ModelSpec, SamplerConfig

MANIFEST = "bundle.json"


@dataclass
class Bundle:
    """A saved analysis, on disk."""

    root: Path
    manifest: dict[str, Any]

    @property
    def spec(self) -> ModelSpec:
        return ModelSpec.model_validate(self.manifest["spec"])

    @property
    def sampler(self) -> SamplerConfig:
        return SamplerConfig.model_validate(self.manifest["sampler"])

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self.manifest["rows"]

    @property
    def stan_source(self) -> str:
        return (self.root / "model.stan").read_text(encoding="utf-8")


def save(
    root: Path,
    spec: ModelSpec,
    rows: list[dict[str, Any]],
    sampler: SamplerConfig,
    stan_source: str,
    summary: dict[str, Any] | None = None,
    warnings: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> Bundle:
    """Write a bundle to ``root``, creating it if needed."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "model.stan").write_text(stan_source, encoding="utf-8")

    manifest: dict[str, Any] = {
        "bundle_version": 1,
        "schema_version": SCHEMA_VERSION,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "spec": spec.model_dump(mode="json"),
        "rows": rows,
        # The seed must be the *resolved* one, not the possibly-empty request.
        "sampler": sampler.model_dump(mode="json"),
        "toolchain": _safe_fingerprint(),
        "warnings": warnings or [],
    }
    if summary is not None:
        manifest["summary"] = summary
    if extra:
        manifest.update(extra)

    (root / MANIFEST).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return Bundle(root=root, manifest=manifest)


def load(root: Path) -> Bundle:
    manifest = json.loads((root / MANIFEST).read_text(encoding="utf-8"))
    version = manifest.get("schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(
            f"This analysis was saved with specification version {version}, but "
            f"this version of the tool reads version {SCHEMA_VERSION}."
        )
    return Bundle(root=root, manifest=manifest)


def to_zip(root: Path, destination: Path) -> Path:
    """Pack a bundle directory into a single downloadable archive."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path != destination:
                archive.write(path, path.relative_to(root))
    return destination


def from_zip(archive: Path, root: Path) -> Bundle:
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(root)
    return load(root)


def _safe_fingerprint() -> dict[str, str]:
    try:
        return toolchain_fingerprint()
    except Exception as exc:  # a bundle is still worth saving without CmdStan
        return {"error": str(exc)}
