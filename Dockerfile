FROM python:3.13-slim

ENV POETRY_VIRTUALENVS_CREATE=false \
    POETRY_CACHE_DIR='/var/cache/pypoetry' \
    POETRY_HOME='/usr/local' \
    POETRY_VERSION=1.8.4

RUN apt-get update \
    && apt-get install -y git curl \
    && curl -sSL https://install.python-poetry.org | python3 -

WORKDIR /app
COPY poetry.lock pyproject.toml /app/

RUN poetry install --no-interaction --no-ansi