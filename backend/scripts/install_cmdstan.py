"""Install the pinned CmdStan; the same as ``pirao install-stan``.

Kept for the Docker build and for checkouts that predate the command.
"""

from pirao.install import install_stan

if __name__ == "__main__":
    raise SystemExit(install_stan())
