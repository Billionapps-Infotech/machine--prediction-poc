# Streamlit + CatBoost friendly
FROM python:3.11-slim

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential curl git libgomp1 graphviz \
    && rm -rf /var/lib/apt/lists/*

# Create user first
RUN useradd -m appuser

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy code
COPY . /app

# ✅ Make /app writable for appuser and create a tmp dir
RUN mkdir -p /app/tmp /app/storage && chown -R appuser:appuser /app
ENV TMPDIR=/app/tmp

USER appuser

EXPOSE 8501

CMD ["streamlit","run","app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]

HEALTHCHECK --interval=30s --timeout=5s --retries=5 \
  CMD curl -f http://localhost:8501/_stcore/health || exit 1
