# Portability option. The application host runs this as a systemd service with a
# plain venv (see deploy/), because there is no container runtime there and adding
# one to run a single read-only process would be more moving parts, not fewer.
# This file exists so the harness can be run anywhere else, and so the runtime
# dependencies are stated explicitly rather than inferred.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Dependencies first, so a code change does not invalidate the dependency layer.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

# The corpus, the SQL checks and the generator are part of the application: the
# service scores against them at request time.
COPY data ./data
COPY sql ./sql
COPY tools ./tools

# No credentials are baked in. DEEPSEEK_API_KEY is supplied at run time, and the
# harness degrades to the deterministic revisions with a clear message if it is
# absent.
ENV GEQ_CORPUS=data/golden

# Run as a non-root user; nothing here needs to write to the image.
RUN useradd --create-home --uid 10001 appuser
USER appuser

EXPOSE 8099
HEALTHCHECK --interval=30s --timeout=5s --start-period=5s --retries=3 \
  CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8099/health', timeout=4)"

CMD ["uvicorn", "geq.api:app", "--host", "0.0.0.0", "--port", "8099", "--no-server-header"]
