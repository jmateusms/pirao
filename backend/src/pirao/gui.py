"""``pirao gui``: the API and the built web app from one local process.

The same launch as ``farofa gui`` and ``faultree gui``: bind 127.0.0.1, try
the default port and fall back to a free one, open the browser.  The Docker
setup (nginx in front of the API) is unchanged; this is the path for a laptop
that has Python and CmdStan but no containers.
"""

from __future__ import annotations

import errno
import os
import socket
import sys
import threading
import webbrowser
from pathlib import Path

DEFAULT_PORT = 8765

_PACKAGE = Path(__file__).resolve().parent


def find_static() -> Path | None:
    """Where the built frontend lives: an override, the package, or a checkout."""
    candidates = [
        os.environ.get("PIRAO_STATIC") or os.environ.get("RELIMCMC_STATIC"),
        _PACKAGE / "gui_static",
        # src/pirao -> backend/src -> backend -> repository root
        _PACKAGE.parents[2] / "frontend" / "dist",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "index.html").is_file():
            return Path(candidate)
    return None


def _free_port(host: str, preferred: int | None) -> int:
    if preferred is not None:
        return preferred
    with socket.socket() as probe:
        try:
            probe.bind((host, DEFAULT_PORT))
            return DEFAULT_PORT
        except OSError as error:
            if error.errno != errno.EADDRINUSE:
                raise
    with socket.socket() as probe:
        probe.bind((host, 0))
        return probe.getsockname()[1]


def serve(
    host: str = "127.0.0.1", port: int | None = None, browser: bool = True
) -> int:
    import uvicorn
    from fastapi.staticfiles import StaticFiles
    from starlette.middleware.trustedhost import TrustedHostMiddleware

    from .api.main import app, get_health

    static = find_static()
    if static is None:
        print(
            "error: the web app is not built.  Run `npm install && npm run build` "
            "in frontend/, or set PIRAO_STATIC to a built copy.",
            file=sys.stderr,
        )
        return 2

    health = get_health()
    if not health["ok"]:
        print(f"warning: {health['error']}  {health.get('hint', '')}", file=sys.stderr)

    if host in ("127.0.0.1", "localhost", "::1"):
        # A page on another site can reach a local port; refusing other Host
        # headers keeps DNS rebinding from turning that into access.
        app.add_middleware(
            TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost", "[::1]"]
        )
    # Mounted last, so every /api route above still wins.
    app.mount("/", StaticFiles(directory=static, html=True), name="gui")

    port = _free_port(host, port)
    shown = "localhost" if host in ("127.0.0.1", "::1") else host
    url = f"http://{shown}:{port}/"
    print(f"pirao gui at {url}  (Ctrl+C to stop)")
    if browser:
        threading.Timer(0.6, webbrowser.open, args=(url,)).start()
    uvicorn.run(app, host=host, port=port, log_level="warning")
    return 0
