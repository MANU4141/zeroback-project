# OOTD-AI Project

### _AI가 추천하는 오늘의 스타일_

2025년 가천x세종 연합 학술제 팀 제로백의 산출물인 OOTD-AI는 예시 스타일 이미지/사용자의 위치에 따른 날씨/사용자의 요구사항을 바탕으로 의류를 추천하는 서비스입니다.

## Quick Start

### Docker를 사용한 배포 (권장)

```bash
# 1. 프로젝트 클론
git clone https://github.com/MANU4141/zeroback-project.git
cd zeroback-project

# 2. 환경 변수 설정
cp backend/.env.example backend/.env
# backend/.env 파일에서 API 키들을 설정하세요

# 3. React 앱 빌드
cd frontend/client
npm install
npm run build
cd ../..

# 4. Docker로 배포
docker-compose up -d

# 5. 서비스 접속
# Frontend: http://localhost
# Backend API: http://localhost/api
# Swagger UI: http://localhost/api/docs
```

### 개발 환경 설정

```bash
# 개발용 Docker Compose 실행
docker-compose -f docker-compose.dev.yml up -d

# 서비스 접속
# Frontend: http://localhost:3000
# Backend: http://localhost:5000
```

## Features

- **웹 인터페이스**: React 기반 UI
- **실시간 날씨 정보**: 한국 기상청 API 연동
- **스타일 선택**: 버튼과 텍스트를 통한 사용자 스타일 입력
- **이미지 분석**: YOLO 기반 다중 이미지 AI 분석
- **AI 추천**: 다양한 요소를 종합한 의류 추천
- **LLM 연동**: Gemini API를 통한 개인화된 스타일 추천 문구 생성

## Tech Stack

### Backend

- **Python 3.9** - 백엔드 언어
- **Flask** - 웹 프레임워크
- **YOLOv11** - AI 이미지 분석
- **PyTorch** - 딥러닝 프레임워크
- **OpenCV** - 이미지 처리
- **Google Gemini API** - LLM 텍스트 생성

### Frontend

- **React** - 웹 앱 프론트엔드

### Infrastructure

- **Docker & Docker Compose** - 컨테이너화 및 배포
- **Nginx** - 웹 서버 및 리버스 프록시
- **한국 기상청 API** - 실시간 날씨 데이터

## API Documentation

서비스 실행 후 Swagger UI에서 API 문서를 확인할 수 있습니다:

- **프로덕션**: http://localhost/api/docs
- **개발환경**: http://localhost:5000/apidocs

### 주요 엔드포인트

- `POST /api/recommend` - AI 기반 의상 추천
- `GET /api/health` - 서비스 상태 확인
- `GET /api/debug/ai-status` - AI 모델 상태 확인
- `GET /api/debug/weather-test` - 날씨 API 테스트

## 환경 설정

### 필수 API 키 설정

`backend/.env` 파일에 다음 API 키들을 설정해야 합니다:

```env
# 기상청 API 키 (필수)
WEATHER_API_KEY_ENCODE=your_encoded_api_key
WEATHER_API_KEY_DECODE=your_decoded_api_key

# Gemini API 키 (아직 미완성)
GEMINI_API_KEY=your_gemini_api_key
```

#### API 키 발급 방법:

1. **기상청 API**: [공공데이터포털](https://www.data.go.kr)에서 기상청\_단기예보 조회서비스 신청
2. **Gemini API**: [Google AI Studio](https://aistudio.google.com)에서 API 키 발급
