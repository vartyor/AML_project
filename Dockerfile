# ──────────────────────────────────────────────────────
# HF Spaces Docker SDK
# - 포트: 7860 (HF Spaces 필수)
# - 유저: non-root (uid=1000)
# ──────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    curl \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# CPU-only PyTorch (이미지 크기 최소화)
RUN pip install --no-cache-dir \
    torch==2.4.0+cpu \
    torchvision==0.19.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# torch-geometric
RUN pip install --no-cache-dir torch-geometric==2.5.2

# 나머지 의존성 (COPY로 직접 복사 → Xet 마운트 우회)
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# 앱 소스 복사
COPY app.py .
COPY analysis/ ./analysis/
COPY loaders/ ./loaders/
COPY models/ ./models/
COPY reporters/ ./reporters/
COPY knowledge/ ./knowledge/
COPY knowledge_base/ ./knowledge_base/
COPY tools/ ./tools/
COPY neo4j_config.py .
COPY NanumGothic*.ttf ./

# 모델 파일
COPY fraud_model.pkl .
COPY gnn_model.pth .

# Streamlit 설정
RUN mkdir -p /app/.streamlit
COPY .streamlit/ ./.streamlit/

# HF Spaces non-root 유저 설정
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# 포트 (HF Spaces Docker SDK 필수)
EXPOSE 7860

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:7860/_stcore/health || exit 1

# 실행
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=7860", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
