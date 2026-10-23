FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODEL_CACHE=/models \
    DATA_DIR=/data \
    HF_HUB_OFFLINE=0

WORKDIR /srv
COPY requirements.txt requirements-dev.txt ./
RUN pip install -r requirements-dev.txt

# Bake the model into the image so the container runs fully offline.
ARG EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBED_MODEL=${EMBED_MODEL}
RUN python -c "from fastembed import TextEmbedding as T; import os; \
list(T(os.environ['EMBED_MODEL'], cache_dir='/models').embed(['warmup']))"
ENV HF_HUB_OFFLINE=1

COPY app ./app
COPY tests ./tests
COPY pytest.ini ./

VOLUME ["/data"]
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
