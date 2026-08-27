# Container image for an agent served by wardhook-core.
#
# Build from the repository root so the package source is in context:
#   docker build -f packages/wardhook-core/Dockerfile -t my-agent .
#
# Run, supplying the agent to serve and a provider key from the environment:
#   docker run --rm -p 8000:8000 \
#     -e ANTHROPIC_API_KEY \
#     -e WARDHOOK_TARGET=myapp.agents:support_agent \
#     -v "$PWD/myapp:/app/myapp:ro" \
#     my-agent
#
# No credential is baked into the image. Keys arrive at runtime through the
# environment, so the same image is safe to push to a registry.

# ---- build stage -----------------------------------------------------------
FROM python:3.12-slim AS build

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /build

# Copy only what the build needs, so edits to tests or docs do not bust the
# dependency layer.
COPY packages/wardhook-core/pyproject.toml packages/wardhook-core/README.md \
     packages/wardhook-core/LICENSE ./
COPY packages/wardhook-core/src ./src

RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip \
    && /opt/venv/bin/pip install .

# ---- runtime stage ---------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    WARDHOOK_HOST=0.0.0.0 \
    WARDHOOK_PORT=8000

# Run as an unprivileged user. An agent executes model-chosen tool calls, so
# it should never hold more privilege than it strictly needs.
RUN useradd --create-home --uid 10001 wardhook

COPY --from=build /opt/venv /opt/venv

WORKDIR /app
USER wardhook

EXPOSE 8000

# Uses the /health endpoint the serve app exposes.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,os,sys; \
sys.exit(0 if urllib.request.urlopen(f'http://127.0.0.1:{os.getenv(\"WARDHOOK_PORT\",\"8000\")}/health', timeout=2).status==200 else 1)"

# WARDHOOK_TARGET names the agent to serve, as module:attribute.
ENV WARDHOOK_TARGET=agent:agent
ENTRYPOINT ["sh", "-c", "exec wardhook serve \"$WARDHOOK_TARGET\" --host \"$WARDHOOK_HOST\" --port \"$WARDHOOK_PORT\""]
