# Backend Dockerfile
FROM python:3.9-slim

# 시스템 의존성 설치 (OpenCV용 필수 라이브러리)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libglib2.0-0 \
    libgl1 \
    libglvnd0 \
    libgl1-mesa-dri \
    libglu1-mesa \
    libglx-mesa0 \
    curl \
    && rm -rf /var/lib/apt/lists/*
# 작업 디렉토리 설정
WORKDIR /app

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드 복사
COPY . .

# 환경변수 설정
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 5000

# 헬스체크 추가
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:5000/api/health || exit 1

# 애플리케이션 실행 (run.py 진입점 사용)
CMD ["python", "run.py"]