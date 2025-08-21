# Backend API Server

## 📋 목차
- [프로젝트 개요](#프로젝트-개요)
- [시스템 아키텍처](#시스템-아키텍처)
- [설치 및 실행](#설치-및-실행)
- [API 문서](#api-문서)
- [디렉토리 구조](#디렉토리-구조)
- [핵심 기능](#핵심-기능)
- [환경 설정](#환경-설정)
- [Docker 배포](#docker-배포)
- [테스트](#테스트)
- [개발자 가이드](#개발자-가이드)

## 🎯 프로젝트 개요

OOTD-AI 백엔드는 AI 기반 패션 추천 시스템의 핵심 API 서버입니다.
사용자가 업로드한 의류 이미지를 분석하고, 현재 날씨 정보와 사용자 선호도를 고려하여 개인화된 패션 추천을 제공합니다.

### 주요 특징
- **AI 기반 이미지 분석**: YOLO + ResNet50을 활용한 의류 속성 추출
- **실시간 날씨 연동**: 기상청 API를 통한 날씨 기반 추천
- **LLM 통합**: Google Gemini API를 활용한 자연어 추천 생성
- **RESTful API**: Flask 기반의 확장 가능한 API 서버
- **Docker 지원**: 컨테이너화된 배포 환경

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐      ┌─────────────────┐      ┌─────────────────┐
│   Frontend      │────> │   Backend API   │ <──> │   AI Models     │
│   (React)       │      │     (Flask)     │      │ (YOLO/ResNet)   │
└─────────────────┘      └───────┬─────────┘      └─────────────────┘
                                 │
                     ┌───────────┼───────────┐
                     │           │           │
               ┌─────▼─────┐ ┌───▼───┐ ┌─────▼────┐
               │ Weather   │ │  LLM  │ │ Database │
               │   API     │ │(Gemini)│ │ (Images) │
               └───────────┘ └───────┘ └──────────┘
```

## 🚀 설치 및 실행

### 필수 요구사항
- Python 3.9+
- CUDA (GPU 사용 시, 선택사항)
- 최소 4GB RAM

### 1. 환경 설정
```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화 (Windows)
venv\Scripts\activate

# 가상환경 활성화 (Linux/Mac)
source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt
```

### 2. 환경변수 설정
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가합니다. (`config.py` 참조)

```env
# 기상청 API 키
WEATHER_API_KEY_DECODE=your_weather_api_key_decoded

# Google Gemini API 키
GEMINI_API_KEY=your_gemini_api_key

# Flask 설정
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000
FLASK_DEBUG=True

# CORS 설정 (프론트엔드 주소에 맞게 수정)
CORS_ORIGINS=http://localhost:3000,[http://127.0.0.1:3000](http://127.0.0.1:3000)
```

### 3. 모델 파일 준비
다음 AI 모델 파일들이 `models/` 디렉토리에 있어야 합니다:
- `YOLOv11_large.pt`: YOLO 객체 탐지 모델
- `ResNet50.pth`: ResNet 속성 분류 모델

### 4. 서버 실행
```bash
# 개발 서버 실행
python run.py
```

서버가 성공적으로 실행되면 `http://localhost:5000`에서 접근 가능합니다.

## 📚 API 문서

### Health Check
서버 상태 및 AI 모델 로드 상태를 확인합니다.
```http
GET /api/health
```
**응답 예시:** (`app/routes.py` 기준)
```json
{
    "db_images_count": 1000,
    "models_initialized": true,
    "resnet_model_loaded": true,
    "status": "OK",
    "timestamp": "2024-08-21T14:30:00.123456",
    "yolo_model_loaded": true
}
```

### 의상 추천
날씨, 사용자 요청, 이미지를 기반으로 의상을 추천합니다.
```http
POST /api/recommend
Content-Type: multipart/form-data
```
**파라미터:**
- `data` (string, required): JSON 형태의 요청 데이터
- `images` (file, optional): 분석할 의류 이미지 (여러 개 가능)

**요청 데이터(`data` 필드) 예시:**
```json
{
    "location": "서울",
    "latitude": 37.5665,
    "longitude": 126.9780,
    "user_request": "캐주얼하면서도 세련된 스타일로 입고 싶어요",
    "style_select": ["캐주얼", "모던"],
    "page": 1,
    "per_page": 3
}
```

**응답 예시:** (`app/routes.py` 기준)
```json
{
    "success": true,
    "weather": {
        "temperature": 22.0,
        "condition": "맑음",
        "humidity": 60,
        "wind_speed": 1.5
    },
    "styling_tip": "오늘처럼 맑은 날씨에 '캐주얼하면서도 세련된 스타일'을 원하시는군요! 화사한 컬러의 맨투맨과 깔끔한 청바지를 매치해 편안하면서도 스타일리시한 룩을 연출해보세요.",
    "recommended_images": [
        {
            "img_path": "image_name_123.jpg",
            "similarity_score": 15.5,
            "label": { "category": "티셔츠", "style": "캐주얼", "color": "블루" }
        }
    ],
    "suggested_items": ["맨투맨", "청바지", "스니커즈"],
    "pagination": {
        "current_page": 1,
        "total_pages": 10
    },
    "debug_info": {
        "ai_analysis": { "category": ["티셔츠"], "style": ["캐주얼"], "color": ["블루"] },
        "ai_debug_details": [],
        "weather_fallback_used": false
    }
}
```

### Swagger UI
개발 중에는 `http://localhost:5000/apidocs`에서 인터랙티브 API 문서를 확인할 수 있습니다.

## 📁 디렉토리 구조

```
backend_gemini/
├── app/                  # Flask 애플리케이션 패키지
│   ├── __init__.py       # Flask 앱 팩토리
│   ├── routes.py         # API 라우트 정의
│   ├── services.py       # 비즈니스 로직 서비스
│   ├── schemas.py        # API 스키마 정의
│   ├── utils.py          # 유틸리티 함수들
│   ├── weather.py        # 날씨 API 클라이언트
│   ├── ai/               # AI 모델 관련 모듈
│   │   ├── yolo_classification.py         # YOLO 모델 래퍼
│   │   └── resnet_multi_classification.py # ResNet 모델 래퍼
│   ├── llm/              # LLM 관련 모듈
│   │   └── gemini_prompt_utils.py       # Gemini API 유틸리티
│   └── recommender/      # 추천 시스템
│       ├── final_recommender.py         # 최종 추천 로직
│       └── style_mappings.py            # 스타일 키워드 매핑
├── DATA/
│   ├── images/           # 의류 이미지 데이터베이스
│   └── labels/           # 라벨 데이터
├── models/
│   ├── YOLOv11_large.pt  # YOLO 모델
│   └── ResNet50.pth      # ResNet 모델
├── config.py             # 설정 파일
├── run.py                # 서버 실행 진입점
├── requirements.txt      # Python 의존성
├── Dockerfile            # Docker 이미지 빌드 파일
└── swagger.yaml          # Swagger API 명세
```

## ⚙️ 핵심 기능

### 1. AI 이미지 분석
- **YOLO 객체 탐지**: 의류 아이템 식별 및 위치 탐지
- **ResNet 속성 분류**: 카테고리, 스타일, 색상 등 다중 속성 추출
- **다중 이미지 처리**: 여러 이미지를 입력받아 종합적인 속성 요약

### 2. 날씨 기반 추천
- **실시간 날씨 정보**: 기상청 단기예보 API 연동
- **위치 기반 서비스**: 위경도 좌표를 통한 정확한 날씨 데이터 조회
- **날씨 규칙**: 온도, 날씨 상태에 따른 추천 아이템 필터링

### 3. 개인화 추천 시스템
- **사용자 요청 분석**: Gemini LLM을 활용해 사용자의 자연어 요청을 키워드, 색상, 스타일 등으로 구조화
- **스타일 선호도**: 텍스트 분석을 통해 "귀엽게", "깔끔하게" 등 키워드에 가중치를 부여
- **종합 점수화**: 날씨, AI 분석, 사용자 선호도를 종합하여 DB 이미지에 점수를 매겨 최적의 아이템 추천

## 🔧 환경 설정 (`config.py`)

`config.py` 파일은 모델 및 데이터 경로, API 키, 서버 설정, 클래스 매핑 등 애플리케이션의 핵심 설정을 관리합니다.

```python
class Config:
    # 기본 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로
    MODEL_PATHS = {
        "yolo": os.path.join(BASE_DIR, "models", "YOLOv11_large.pt"),
        "resnet": os.path.join(BASE_DIR, "models", "ResNet50.pth"),
    }
    
    # 데이터 경로
    IMAGE_DIR = os.path.join(BASE_DIR, "DATA", "images")
    LABELS_DIR = os.path.join(BASE_DIR, "DATA", "labels")
    
    # 서버 설정
    FLASK_RUN_HOST = "0.0.0.0"
    FLASK_RUN_PORT = 5000
    FLASK_DEBUG = True
    
    # 클래스 매핑 (학습된 모델 기준)
    CLASS_MAPPINGS = {
        "category": ["탑", "블라우스", "티셔츠", ...],
        "style": ["레트로", "로맨틱", "스트리트", ...],
        "color": ["골드", "그레이", "그린", ...]
    }
```

## 🐳 Docker 배포

### Docker 이미지 빌드
```bash
docker build -t ootd-backend .
```

### Docker 컨테이너 실행
`.env` 파일의 내용을 환경변수로 전달하여 실행합니다.
```bash
docker run -p 5000:5000 \
  -e WEATHER_API_KEY_DECODE=your_key \
  -e GEMINI_API_KEY=your_key \
  ootd-backend
```

### Docker Compose 사용
```bash
# 개발 환경
docker-compose -f docker-compose.dev.yml up

# 프로덕션 환경
docker-compose up
```

## 🧪 테스트

### API 테스트 (cURL)

```bash
# 헬스 체크
curl http://localhost:5000/api/health

# 추천 API 테스트 (이미지 없이)
curl -X POST -H "Content-Type: multipart/form-data" \
-F 'data={"location": "서울", "latitude": 37.5665, "longitude": 126.9780, "user_request": "캐주얼한 스타일"}' \
http://localhost:5000/api/recommend
```

## 👨‍💻 개발자 가이드

### 새로운 API 엔드포인트 추가

1.  **스키마 정의** (`app/schemas.py`): `flasgger`가 인식할 수 있는 API 명세를 작성합니다.
2.  **서비스 로직 구현** (`app/services.py`): 실제 비즈니스 로직을 함수로 구현합니다.
3.  **라우트 등록** (`app/routes.py`): Flask의 `@app.route()` 데코레이터를 사용하여 엔드포인트를 등록하고 서비스 함수를 연결합니다.

### 로깅 설정

기본 로깅 설정은 `app/__init__.py`에 정의되어 있으며, Flask의 `current_app.logger`를 통해 애플리케이션 전역에서 로그를 기록할 수 있습니다.
```python
# 예시: routes.py 에서 로거 사용
from flask import current_app

current_app.logger.info("이것은 정보 로그입니다.")
current_app.logger.error("이것은 에러 로그입니다.")
```