# ── Imagen base ────────────────────────────────────────────────────────────
FROM python:3.10-slim

# ── Dependencias de sistema ─────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgles2 \
    libegl1 \
    && rm -rf /var/lib/apt/lists/*

# ── Usuario no-root (requerido por Hugging Face Spaces) ────────────────────
RUN useradd -m -u 1000 appuser

# ── Directorio de trabajo ───────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python ─────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Código de la aplicación ─────────────────────────────────────────────────
COPY --chown=appuser:appuser . .

# ── Cambiar a usuario no-root ───────────────────────────────────────────────
USER appuser

# ── Puerto y variable de entorno ────────────────────────────────────────────
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

# ── Inicio ──────────────────────────────────────────────────────────────────
CMD ["python", "-u", "app.py"]
