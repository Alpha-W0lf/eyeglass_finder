# 1. Base image with Python and essential system packages
FROM python:3.12-slim-bullseye as base

ARG GIT_COMMIT_HASH
ENV GIT_COMMIT_HASH=${GIT_COMMIT_HASH}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry'

WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libomp-dev \
    libgl1 \
    libcairo2 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --upgrade pip && pip install cmake==3.28.3

# ----------------------------------------------------------------------
# 2. Dependency builder stages
# ----------------------------------------------------------------------
FROM base as deps_base
RUN pip install --no-cache-dir poetry
COPY poetry.lock pyproject.toml ./

FROM deps_base as prod_deps
RUN poetry install --no-root --only main

FROM prod_deps as dev_deps
RUN poetry install --no-root --with dev

# ----------------------------------------------------------------------
# 3. Final application stage
# ----------------------------------------------------------------------
FROM base as app
ENV STAGE=app
COPY --from=prod_deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src ./src
COPY scripts ./scripts
COPY models ./models
COPY config ./config
CMD ["python", "scripts/process_data.py"]

# ----------------------------------------------------------------------
# 4. Test stage with dev dependencies
# ----------------------------------------------------------------------
FROM base as test
ENV STAGE=test
COPY --from=dev_deps /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY src ./src
COPY tests ./tests
