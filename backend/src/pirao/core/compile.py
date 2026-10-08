"""Compile generated Stan source, with an on-disk cache.

Compilation is the only wait a user actually experiences.  Sampling one of
these models takes milliseconds; a cold ``stanc`` + ``g++`` pass takes tens of
seconds.  So the cache is not an optimisation, it is the feature that makes the
tool feel responsive -- and because bound and hyperparameter *values* are Stan
data rather than source, the common interactions (nudging a hyperparameter,
tightening a bound, editing the table) all hit it.

The cache key deliberately includes the toolchain, not just the source.  Keying
on source alone is a latent correctness bug: upgrade CmdStan and every stale
binary is silently reused, built by the previous ``stanc``, with nothing to
indicate it.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from .render import RenderedModel

#: Bump when a change to this package alters the meaning of generated source
#: without altering the text (for example, a change to how data is assembled).
GENERATOR_VERSION = "1"


class ToolchainError(RuntimeError):
    """CmdStan is missing or unusable."""


def default_cache_dir() -> Path:
    root = os.environ.get("PIRAO_CACHE_DIR") or os.environ.get("RELIMCMC_CACHE_DIR")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pirao" / "models"


def toolchain_fingerprint() -> dict[str, str]:
    """Identify the compiler stack, for the cache key and the run bundle."""
    try:
        import cmdstanpy
    except ImportError as exc:  # pragma: no cover - import guarded at call sites
        raise ToolchainError(
            "cmdstanpy is not installed.  Install the backend requirements first."
        ) from exc

    try:
        version = cmdstanpy.cmdstan_version()
        path = cmdstanpy.cmdstan_path()
    except Exception as exc:
        raise ToolchainError(
            "CmdStan was not found.  Run `python scripts/install_cmdstan.py` "
            "to install it, or set the CMDSTAN environment variable if it is "
            "already installed elsewhere."
        ) from exc

    return {
        "cmdstan_version": ".".join(str(v) for v in version) if version else "unknown",
        "cmdstan_path": str(path),
        "cmdstanpy_version": cmdstanpy.__version__,
        "generator_version": GENERATOR_VERSION,
        "cxxflags": os.environ.get("STAN_CXXFLAGS", ""),
        "stancflags": os.environ.get("STANCFLAGS", ""),
    }


def cache_key(source: str, fingerprint: dict[str, str] | None = None) -> str:
    fp = fingerprint if fingerprint is not None else toolchain_fingerprint()
    payload = "\n".join(
        [
            source,
            fp["cmdstan_version"],
            fp["cmdstanpy_version"],
            fp["generator_version"],
            fp["cxxflags"],
            fp["stancflags"],
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@dataclass(frozen=True)
class CompiledModel:
    key: str
    stan_file: Path
    exe_file: Path
    model: object  # cmdstanpy.CmdStanModel
    was_cached: bool


def _lock(path: Path):
    """A crude cross-process lock: two browser tabs must not race on one build."""
    import fcntl

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w")
    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
    return handle


def compile_model(
    rendered: RenderedModel,
    cache_dir: Path | None = None,
    force: bool = False,
) -> CompiledModel:
    """Compile (or reuse) the binary for ``rendered``.

    The generated ``.stan`` is written exactly once and never touched again:
    cmdstanpy recompiles on mtime change, so rewriting an identical file on a
    cache hit would throw away the cache.
    """
    from cmdstanpy import CmdStanModel

    fingerprint = toolchain_fingerprint()
    key = cache_key(rendered.source, fingerprint)
    root = (cache_dir or default_cache_dir()) / key
    stan_file = root / "model.stan"
    exe_file = root / "model"

    handle = _lock(root.parent / f".{key}.lock")
    try:
        cached = exe_file.exists() and stan_file.exists() and not force
        if not cached:
            if force and root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True, exist_ok=True)
            stan_file.write_text(rendered.source, encoding="utf-8")
            (root / "spec.json").write_text(
                rendered.spec.model_dump_json(indent=2), encoding="utf-8"
            )
            (root / "toolchain.json").write_text(
                json.dumps(fingerprint, indent=2), encoding="utf-8"
            )
            model = CmdStanModel(stan_file=str(stan_file))
        else:
            model = CmdStanModel(exe_file=str(exe_file), stan_file=str(stan_file))
    finally:
        handle.close()

    return CompiledModel(
        key=key,
        stan_file=stan_file,
        exe_file=exe_file,
        model=model,
        was_cached=cached,
    )


def is_cached(rendered: RenderedModel, cache_dir: Path | None = None) -> bool:
    key = cache_key(rendered.source)
    return ((cache_dir or default_cache_dir()) / key / "model").exists()
