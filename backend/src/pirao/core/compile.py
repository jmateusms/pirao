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
import math
import os
import re
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

from .render import RenderedModel

#: Bump when a change to this package alters the meaning of generated source
#: without altering the text (for example, a change to how data is assembled).
GENERATOR_VERSION = "1"

#: Ceiling for the binary cache.  Each structure costs 10 to 40 MB, so this
#: holds some 50 to 200 of them; ``PIRAO_CACHE_MAX_MB`` overrides it and ``0``
#: turns the limit off.
DEFAULT_CACHE_MAX_MB = 2048

#: Entries used this recently are never evicted, even over the limit: a run
#: that has just compiled (or reused) a binary samples it right afterwards,
#: outside the build lock, and must not find the executable gone.
EVICTION_GRACE_SECONDS = 15 * 60

_ENTRY_NAME = re.compile(r"^[0-9a-f]{16}$")
_LAST_USED = ".last-used"


class ToolchainError(RuntimeError):
    """CmdStan is missing or unusable."""


def default_cache_dir() -> Path:
    root = os.environ.get("PIRAO_CACHE_DIR") or os.environ.get("RELIMCMC_CACHE_DIR")
    if root:
        return Path(root)
    base = os.environ.get("XDG_CACHE_HOME") or (Path.home() / ".cache")
    return Path(base) / "pirao" / "models"


def cache_max_bytes() -> int | None:
    """The cache ceiling in bytes, or None when the limit is off."""
    raw = (os.environ.get("PIRAO_CACHE_MAX_MB") or "").strip()
    try:
        mb = float(raw) if raw else DEFAULT_CACHE_MAX_MB
    except ValueError:
        mb = DEFAULT_CACHE_MAX_MB
    if math.isnan(mb):
        mb = DEFAULT_CACHE_MAX_MB
    return None if mb <= 0 or math.isinf(mb) else int(mb * 1024 * 1024)


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


def _try_lock(path: Path):
    """The same lock without waiting: None when another process holds it."""
    import fcntl

    handle = path.open("w")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        handle.close()
        return None
    return handle


def _last_used(entry: Path) -> float:
    marker = entry / _LAST_USED
    if marker.exists():
        return marker.stat().st_mtime
    # Entries built before the marker existed: the newest file is the build.
    return max((p.stat().st_mtime for p in entry.iterdir()), default=0.0)


def _entry_bytes(entry: Path) -> int:
    return sum(p.stat().st_size for p in entry.rglob("*") if p.is_file())


def prune_cache(
    cache_dir: Path,
    max_bytes: int | None,
    keep: frozenset[str] = frozenset(),
    grace_seconds: float = EVICTION_GRACE_SECONDS,
) -> list[str]:
    """Evict least recently used binaries until the cache fits ``max_bytes``.

    Entries named in ``keep``, used within ``grace_seconds``, or whose build
    lock another process holds are skipped, so the cache can stay over the
    limit until they age out.  Lock files are left in place: unlinking one
    while another process waits on it would let two builds of the same key
    run at once.  Returns the evicted keys.
    """
    if max_bytes is None or not cache_dir.is_dir():
        return []
    entries = []
    for entry in cache_dir.iterdir():
        if entry.is_dir() and _ENTRY_NAME.fullmatch(entry.name):
            entries.append((_last_used(entry), entry.name, _entry_bytes(entry)))
    total = sum(size for _, _, size in entries)
    now = time.time()
    evicted = []
    for used, key, size in sorted(entries):
        if total <= max_bytes:
            break
        if key in keep or now - used < grace_seconds:
            continue
        handle = _try_lock(cache_dir / f".{key}.lock")
        if handle is None:
            continue
        try:
            # A run may have reused the entry between the scan and the lock.
            if time.time() - _last_used(cache_dir / key) < grace_seconds:
                continue
            shutil.rmtree(cache_dir / key, ignore_errors=True)
        finally:
            handle.close()
        total -= size
        evicted.append(key)
    return evicted


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
    cache_dir = cache_dir or default_cache_dir()
    root = cache_dir / key
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
        # A separate marker, because touching the executable or the source
        # would change what cmdstanpy compares to decide on a rebuild.
        (root / _LAST_USED).touch()
    finally:
        handle.close()

    if not cached:
        # Eviction must never fail a run that already has its binary.
        try:
            prune_cache(cache_dir, cache_max_bytes(), keep=frozenset({key}))
        except (OSError, ValueError):
            pass

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
