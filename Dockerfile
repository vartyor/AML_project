# ──────────────────────────────────────────────────────
# Stage 1: 빌드 스테이지 (의존성 설치)
# ──────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /app

# 시스템 패키지 설치 (PyTorch Geometric 등 C 확장 빌드용)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip 업그레이드
RUN pip install --upgrade pip

# CPU-only PyTorch 먼저 설치 (이미지 크기 최소화)
RUN pip install --no-cache-dir \
    torch==2.2.2+cpu \
    torchvision==0.17.2+cpu \
    --index-url https://download.pytorch.org/whl/cpu

# torch-geometric 설치
RUN pip install --no-cache-dir torch-geometric==2.5.2

# 나머지 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ──────────────────────────────────────────────────────
# Stage 2: 런타임 스테이지
# ──────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

WORKDIR /app

# 런타임 시스템 패키지 (최소화)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# 빌드 스테이지에서 Python 패키지 복사
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 앱 소스 코드 복사
COPY app.py .
COPY analysis/ ./analysis/
COPY loaders/ ./loaders/
COPY models/ ./models/
COPY reporters/ ./reporters/
COPY knowledge/ ./knowledge/
COPY tools/ ./tools/

# 모델 파일 복사 (존재하는 경우)
COPY fraud_model.pkl* ./
COPY gnn_model.pth* ./

# 데이터셋 복사 (Streamlit Cloud에서는 Git LFS 활용 권장)
COPY dataset/ ./dataset/

# 지식 베이스 복사
COPY knowledge_base/ ./knowledge_base/

# Streamlit 설정
RUN mkdir -p /app/.streamlit
COPY .streamlit/ ./.streamlit/ 2>/dev/null || true

# 폰트 파일 복사 (한글 렌더링)
COPY NanumGothic*.ttf ./

# 포트 노출
EXPOSE 8501

# 헬스체크
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# 앱 실행
ENTRYPOINT ["streamlit", "run", "app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.headless=true", \
    "--browser.gatherUsageStats=false"]
