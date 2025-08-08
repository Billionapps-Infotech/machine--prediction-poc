# Streamlit + CatBoost friendly
FROM python:3.11-slim

# System deps (OpenMP for CatBoost), plus graphviz if you render diagrams
RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential curl git libgomp1 graphviz \
    && rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m appuser
WORKDIR /app

# Install Python deps first for better Docker layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the rest of your app
COPY . /app

# Streamlit runtime config
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    PYTHONUNBUFFERED=1

USER appuser
EXPOSE 8501

# Start Streamlit (change path if your file isn’t app.py at repo root)
CMD ["streamlit","run","app.py", "--server.port=8501",  "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1