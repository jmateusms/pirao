FROM python:3.11-slim

# CmdStan is compiled from source at image build time, so a container start is
# instant and no user ever waits on the toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

# The server compiles and runs Stan programs, so it gets no root privileges.
RUN useradd --create-home --uid 10001 pirao \
    && mkdir -p /data && chown pirao:pirao /data

WORKDIR /app
# The package metadata reads README.md, and LICENSE/NOTICE ship with it.
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[api,export]"

USER pirao

# The version is pinned deliberately: resolving "latest" needs a call to
# api.github.com that many corporate proxies block, and an unpinned toolchain
# would silently change what the compile cache is keyed on.  Installed as the
# user that runs the server, into its home.
RUN python scripts/install_cmdstan.py

ENV PIRAO_RUNS_DIR=/data/runs \
    PIRAO_CACHE_DIR=/data/model-cache
VOLUME ["/data"]
EXPOSE 8000

# Pre-compile nothing at build time: the cache key includes the toolchain, so
# warming it here would only help if the image and runtime agree exactly.
CMD ["uvicorn", "pirao.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
