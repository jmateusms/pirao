"""Idempotent, version-pinned CmdStan bootstrap.

The pin is mandatory, not cosmetic: ``install_cmdstan()`` without a version
resolves "latest" through ``api.github.com``, which some egress proxies answer
with 403.  Passing an explicit version skips that lookup entirely and fetches
the release tarball directly.
"""

from __future__ import annotations

import os
import sys

CMDSTAN_VERSION = "2.36.0"


def already_installed() -> bool:
    try:
        from cmdstanpy import cmdstan_path
    except ImportError:
        return False
    try:
        path = cmdstan_path()
    except Exception:
        return False
    return bool(path) and CMDSTAN_VERSION in str(path)


def main() -> int:
    if already_installed():
        print(f"CmdStan {CMDSTAN_VERSION} already present; nothing to do.")
        return 0

    from cmdstanpy import install_cmdstan

    print(f"Installing CmdStan {CMDSTAN_VERSION} (first run takes several minutes)...")
    ok = install_cmdstan(version=CMDSTAN_VERSION, cores=os.cpu_count() or 2)
    if not ok:
        print("CmdStan installation failed.", file=sys.stderr)
        return 1
    print(f"CmdStan {CMDSTAN_VERSION} installed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
