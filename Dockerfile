# syntax=docker/dockerfile:1
FROM python:3.12-slim

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    openssh-client \
    && rm -rf /var/lib/apt/lists/*

# Add GitHub SSH key host verification
RUN mkdir -p -m 0700 ~/.ssh && echo "github.com ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOMqqnkVzrm0SdG6UOoqKLsabgH5C9okWi0dh2l9GKJl" >> ~/.ssh/known_hosts && chmod 600 ~/.ssh/known_hosts

# Set working directory
WORKDIR /app

# Copy uv binary from ghcr.io/astral-sh/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy lockfile and configuration files
COPY pyproject.toml uv.lock ./

# Install project dependencies.
# Requires mounting SSH agent for private git dependency access during build.
RUN --mount=type=ssh uv sync --frozen --no-cache

# Copy the rest of the project source code
COPY src ./src

# Expose evaluation service port
EXPOSE 5002

# Run application using virtualenv uvicorn
RUN useradd -m appuser && chown -R appuser /app
USER appuser
CMD ["/app/.venv/bin/uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "5002"]
