FROM python:3.11-slim

WORKDIR /app

# Instalar dependencias primero (capa cacheable).
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e .

# Resto del proyecto (frontend lo sirve el backend en /).
COPY frontend ./frontend
COPY scripts ./scripts

EXPOSE 8000

# El host (Render/Railway/Fly) inyecta $PORT; default 8000 en local.
CMD ["sh", "-c", "uvicorn markowitz_optimizer.api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
