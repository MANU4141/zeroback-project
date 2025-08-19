# OOTD-AI Project

### _AI가 추천하는 오늘의 스타일_

2025년 가천x세종 연합 학술제 팀 제로백의 산출물인 OOTD-AI는 예시 스타일 이미지/사용자의 위치에 따른 날씨/사용자의 요구사항을 바탕으로 의류를 추천하는 서비스입니다.

## Features

- 🌐 **웹 인터페이스** - 직관적인 React 기반 사용자 인터페이스
- 📍 **위치 기반 날씨 정보** - 기상청 단기예보 API 연동
- 🎨 **스타일 입력** - 버튼 선택 및 텍스트를 통한 개인화된 스타일 설정
- 📸 **이미지 분석** - 업로드된 의류 이미지의 AI 기반 스타일 분석
- 🤖 **AI 추천 시스템** - YOLO + ResNet 모델을 활용한 의류 추천
- 💬 **스마트 추천 문구** - Google Gemini API를 통한 자연어 스타일 가이드 생성

## Tech Stack

### Frontend
- **React 19** - 최신 React를 활용한 모던 웹 애플리케이션
- **React Router** - SPA 라우팅 및 페이지 관리
- **Axios** - HTTP 클라이언트를 통한 API 통신

### Backend
- **Flask 3.0** - Python 웹 프레임워크 및 RESTful API 서버
- **Flask-CORS** - Cross-Origin 요청 처리
- **Flasgger** - Swagger UI를 통한 API 문서 자동화

### AI & ML
- **YOLOv11** - 실시간 객체 탐지를 통한 의류 아이템 식별
- **ResNet50** - 의류 속성 분류 (카테고리, 스타일, 색상, 계절)
- **PyTorch** - 딥러닝 모델 추론 프레임워크

### External APIs
- **기상청 단기예보 API** - 실시간 날씨 정보 및 위치 기반 서비스
- **Google Gemini API** - LLM 기반 자연어 스타일 추천 생성

### Infrastructure
- **Docker & Docker Compose** - 컨테이너화된 마이크로서비스 배포
- **Nginx** - 리버스 프록시 및 정적 파일 서빙

## Installation

OOTD-AI는 Docker Compose 환경을 사용하여 어느 환경에서든 즉시 실행 가능합니다.

### Docker로 실행
```bash
# 개발 환경
docker-compose -f docker-compose.dev.yml up

# 프로덕션 환경  
docker-compose up
```

### 로컬에서 실행

#### 백엔드 서버
```bash
cd backend
pip install -r requirements.txt
python run.py
```

#### 프론트엔드 서버
```bash
cd frontend/client
npm install
npm start
```


## Testing (백엔드 테스트 실행)

백엔드 테스트는 `pytest`로 실행할 수 있습니다. 아래 명령어를 참고하세요.

### 전체 테스트 실행
```bash
cd backend
pytest tests/test_api.py -v
```

### 통합 프로토타입 테스트 실행
```bash
cd backend
python tests/prototype_test.py
```

---

## API Documentation

서버 실행 후 `http://localhost:5000/apidocs`에서 Swagger UI를 통해 API 문서를 확인할 수 있습니다.

### 주요 API 엔드포인트
- `GET /api/health` - 서버 및 AI 모델 상태 확인
- `POST /api/recommend` - 스타일 기반 의류 추천
- `GET /api/debug/ai-status` - AI 모델 로드 상태 확인

## AI Models & APIs

### 사용된 AI 모델
- **YOLOv11 Large** - Ultralytics에서 제공하는 최신 객체 탐지 모델
  - 의류 아이템 실시간 탐지 및 바운딩 박스 생성
  - 높은 정확도와 빠른 추론 속도
  
- **ResNet50** - 다중 속성 분류를 위한 커스텀 학습 모델
  - 의류 카테고리 (상의, 하의, 아우터 등)
  - 스타일 속성 (캐주얼, 포멀, 스트릿 등)
  - 색상 및 계절성 분석

### 외부 API 연동
- **기상청 단기예보 서비스**
  - 실시간 날씨 데이터 (온도, 습도, 강수량)
  - 격자 좌표 기반 정확한 지역별 예보
  - 6시간 단위 날씨 변화 예측

- **Google Gemini 1.5 API**
  - 자연어 기반 스타일 추천 문구 생성
  - 사용자 요청과 날씨 조건을 고려한 개인화된 조언
  - 한국어 패션 용어 및 트렌드 반영

## Project Structure

```
├── backend/                # Flask API 서버
│   ├── app/               # 메인 애플리케이션
│   │   ├── ai/           # YOLO & ResNet 모델 래퍼
│   │   ├── llm/          # Gemini API 클라이언트
│   │   ├── routes/       # API 엔드포인트 모듈화
│   │   └── recommender/  # 추천 알고리즘
│   ├── models/            # AI 모델 파일 (.pt, .pth)
│   ├── DATA/              # 의류 이미지 데이터베이스
│   └── requirements.txt   # Python 의존성
├── frontend/              # React 웹 애플리케이션
│   └── client/           # React 소스코드
├── nginx/                 # Nginx 설정 및 리버스 프록시
└── docker-compose.yml     # Docker 서비스 구성
```

## System Requirements

- **Python 3.9+** (백엔드)
- **Node.js 16+** (프론트엔드)
- **Docker & Docker Compose** (배포)
- **CUDA 지원 GPU** (선택사항, AI 모델 가속)
- **최소 4GB RAM** (AI 모델 로드)

---

**Team Zeroback** | 2025 가천x세종 연합 학술제