FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

RUN useradd --create-home --uid 1000 appuser
WORKDIR /home/appuser/app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api ./api
COPY frontend ./frontend
RUN mkdir -p models && chown -R appuser /home/appuser/app

USER appuser

# Bake the 126 MB quantized model into the image so startup needs no download
RUN python -m api.download_model

EXPOSE 8000

CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
