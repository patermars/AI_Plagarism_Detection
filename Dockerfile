FROM public.ecr.aws/docker/library/python:3.11-slim AS builder

WORKDIR /app
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --prefix=/install -r requirements.txt

# ── Final stage ──
FROM public.ecr.aws/docker/library/python:3.11-slim

WORKDIR /app
COPY --from=builder /install /usr/local
COPY distilbert_detector/ distilbert_detector/
COPY inference.py app.py ./
COPY templates/ templates/

EXPOSE 5000
CMD ["python", "app.py"]
