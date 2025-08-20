# Backend Dockerfile
FROM python:3.9-slim

# 시스템 의존성 설치 (OpenCV용 필수 라이브러리 등)
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

# requirements.txt 먼저 복사 및 설치 (레이어 캐싱 활용)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드, 모델, 데이터를 이미지에 복사
COPY . .

# 환경변수 설정
ENV FLASK_ENV=production
ENV PYTHONPATH=/app

# 포트 노출
EXPOSE 5000

# 헬스체크 (docker-compose.yml에서도 정의했지만, 이미지 자체에 포함하는 것이 좋음)
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -sf http://localhost:5000/api/health || exit 1

# 애플리케이션 실행 (uWSGI 사용)
CMD ["uwsgi", "--ini", "uwsgi.ini"]