# ── Base image (Python 3.10 — matches chronoveritas-infer conda env) ──────────
FROM python:3.10-slim

WORKDIR /app

# ── Environment ───────────────────────────────────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # HuggingFace cache → writable temp dir (required for HF Spaces)
    HF_HOME=/tmp/huggingface \
    # Enable fast HF Hub transfers (hf-transfer)
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    # Silence pip root-user warnings inside Docker
    PIP_NO_CACHE_DIR=1

# ── System deps (for vLLM / scipy / faiss build deps) ────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Python deps (inference stack — torch+cu121 via PyTorch index) ─────────────
COPY requirements_space.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements_space.txt

# ── Copy application code ──────────────────────────────────────────────────────
COPY . .

# ── Runtime directories ────────────────────────────────────────────────────────
RUN mkdir -p data/tasks/generated plots training_logs /tmp/huggingface

# ── Non-root user (required by HuggingFace Spaces) ────────────────────────────
RUN useradd -m -u 1000 user
USER user

EXPOSE 7860

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]