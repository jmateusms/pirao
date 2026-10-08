FROM python:3.11-slim

# CmdStan is compiled from source at image build time, so a container start is
# instant and no user ever waits on the toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY src ./src
COPY scripts ./scripts

RUN pip install --no-cache-dir -e ".[api,export]"

# The version is pinned deliberately: resolving "latest" needs a call to
# api.github.com that many corporate proxies block, and an unpinned toolchain
# would silently change what the compile cache is keyed on.
RUN python scripts/install_cmdstan.py

ENV PIRAO_RUNS_DIR=/data/runs \
    PIRAO_CACHE_DIR=/data/model-cache
VOLUME ["/data"]
EXPOSE 8000

# Pre-compile nothing at build time: the cache key includes the toolchain, so
# warming it here would only help if the image and runtime agree exactly.
CMD ["uvicorn", "pirao.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
