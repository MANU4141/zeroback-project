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
- **AI 기반 이미지 분석**: YOLOv11 + ResNet50을 활용한 의류 속성 추출
- **실시간 날씨 연동**: 기상청 API를 통한 날씨 기반 추천
- **LLM 통합**: Google Gemini API를 활용한 자연어 추천 생성
- **RESTful API**: Flask 기반의 확장 가능한 API 서버
- **Docker 지원**: 컨테이너화된 배포 환경

## 🏗️ 시스템 아키텍처

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │────│   Backend API   │────│   AI Models     │
│   (React)       │    │   (Flask)       │    │   (YOLO/ResNet) │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                    ┌─────────┼─────────┐
                    │         │         │
            ┌───────▼───┐ ┌───▼───┐ ┌───▼────┐
            │ Weather   │ │  LLM  │ │Database│
            │    API    │ │(Gemini)│ │Images │
            └───────────┘ └───────┘ └────────┘
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
프로젝트 루트에 `.env` 파일을 생성하고 다음 내용을 추가:

```env
# 기상청 API 키
WEATHER_API_KEY_ENCODE=your_weather_api_key_encoded
WEATHER_API_KEY_DECODE=your_weather_api_key_decoded

# Google Gemini API 키
GEMINI_API_KEY=your_gemini_api_key

# Flask 설정
FLASK_RUN_HOST=0.0.0.0
FLASK_RUN_PORT=5000
FLASK_DEBUG=True

# CORS 설정
CORS_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
```

### 3. 모델 파일 준비
다음 AI 모델 파일들이 `models/` 디렉토리에 있어야 합니다:
- `YOLOv11_large.pt`: YOLO 객체 탐지 모델
- `ResNet50_45.pth`: ResNet 속성 분류 모델

### 4. 서버 실행
```bash
# 개발 서버 실행
python run.py

# 또는 Flask 명령어로 실행
flask --app run.py run --debug
```

서버가 성공적으로 실행되면 `http://localhost:5000`에서 접근 가능합니다.

## 📚 API 문서

### Health Check
```http
GET /api/health
```
서버 상태 및 AI 모델 로드 상태를 확인합니다.

**응답 예시:**
```json
{
  "status": "OK",
  "timestamp": "2024-01-01T12:00:00",
  "models_initialized": true,
  "yolo_model_loaded": true,
  "resnet_model_loaded": true,
  "db_images_count": 1500
}
```

### 의상 추천
```http
POST /api/recommend
Content-Type: multipart/form-data
```

**파라미터:**
- `data` (string, required): JSON 형태의 요청 데이터
- `image` (file, optional): 분석할 의류 이미지

**요청 데이터 예시:**
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

**응답 예시:**
```json
{
  "status": "success",
  "recommendations": [
    {
      "image_path": "/path/to/recommended/image.jpg",
      "score": 0.95,
      "attributes": {
        "category": "티셔츠",
        "style": "캐주얼",
        "color": "블루",
        "season": "봄"
      }
    }
  ],
  "weather_info": {
    "temperature": 22,
    "weather_condition": "맑음",
    "humidity": 60
  },
  "ai_analysis": {
    "detected_items": ["상의", "하의"],
    "dominant_colors": ["블루", "화이트"],
    "style_attributes": ["캐주얼", "모던"]
  }
}
```

### Swagger UI
개발 중에는 `http://localhost:5000/apidocs`에서 인터렙티브 API 문서를 확인할 수 있습니다.

## 📁 디렉토리 구조

```
backend/
├── app/                    # Flask 애플리케이션 패키지
│   ├── __init__.py        # Flask 앱 팩토리
│   ├── routes.py          # API 라우트 정의
│   ├── services.py        # 비즈니스 로직 서비스
│   ├── schemas.py         # API 스키마 정의
│   ├── utils.py           # 유틸리티 함수들
│   ├── weather.py         # 날씨 API 클라이언트
│   ├── ai/               # AI 모델 모듈
│   │   ├── __init__.py
│   │   ├── yolo_multitask.py      # YOLO 모델 래퍼
│   │   └── resnet_multitask.py    # ResNet 모델 래퍼
│   ├── llm/              # LLM 관련 모듈
│   │   ├── __init__.py
│   │   └── gemini_prompt_utils.py # Gemini API 유틸리티
│   └── recommender/      # 추천 시스템
│       ├── __init__.py
│       ├── db_similarity.py       # 유사도 계산
│       └── final_recommender.py   # 최종 추천 로직
├── DATA/                  # 학습 데이터 및 이미지
│   ├── images/           # 의류 이미지 데이터베이스
│   └── labels/           # 라벨 데이터
├── models/               # AI 모델 파일
│   ├── YOLOv11_large.pt  # YOLO 모델
│   └── ResNet50_45.pth   # ResNet 모델
├── tests/                # 테스트 파일
│   ├── __init__.py
│   └── test_api.py       # API 테스트
├── config.py             # 설정 파일
├── run.py                # 서버 진입점
├── requirements.txt      # Python 의존성
├── Dockerfile           # Docker 이미지 빌드 파일
├── swagger.yaml         # Swagger API 문서
└── README.md            # 이 문서
```

## ⚙️ 핵심 기능

### 1. AI 이미지 분석
- **YOLO 객체 탐지**: 의류 아이템 식별 및 위치 탐지
- **ResNet 속성 분류**: 카테고리, 스타일, 색상, 시즌 등 다중 속성 추출
- **이미지 전처리**: 크기 조정, 정규화, 텐서 변환

### 2. 날씨 기반 추천
- **실시간 날씨 정보**: 기상청 단기예보 API 연동
- **위치 기반 서비스**: GPS 좌표를 통한 정확한 날씨 데이터
- **계절성 고려**: 온도, 습도, 날씨 상태에 따른 의류 필터링

### 3. 개인화 추천 시스템
- **스타일 선호도**: 사용자가 선택한 스타일 가중치 적용
- **유사도 계산**: 코사인 유사도 기반 이미지 매칭
- **LLM 통합**: 자연어 요청을 구조화된 추천으로 변환

### 4. 확장 가능한 아키텍처
- **모듈화 설계**: 각 기능별 독립적인 모듈 구조
- **플러그인 시스템**: 새로운 AI 모델 쉽게 추가 가능
- **캐시 시스템**: 이미지 데이터베이스 메모리 캐싱

## 🔧 환경 설정

### config.py 주요 설정

```python
class Config:
    # 기본 경로
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    
    # 모델 경로
    MODEL_PATHS = {
        "yolo": os.path.join(BASE_DIR, "models", "YOLOv11_large.pt"),
        "resnet": os.path.join(BASE_DIR, "models", "ResNet50_45.pth"),
    }
    
    # 데이터 경로
    IMAGE_DIR = os.path.join(BASE_DIR, "DATA", "images")
    LABELS_DIR = os.path.join(BASE_DIR, "DATA", "labels")
    
    # 서버 설정
    FLASK_RUN_HOST = "0.0.0.0"
    FLASK_RUN_PORT = 5000
    FLASK_DEBUG = True
    
    # 클래스 매핑
    CLASS_MAPPINGS = {
        "category": ["탑", "블라우스", "티셔츠", ...],
        "style": ["캐주얼", "포멀", "스트릿", ...],
        "color": ["블랙", "화이트", "레드", ...],
        "season": ["봄", "여름", "가을", "겨울"]
    }
```

## 🐳 Docker 배포

### Docker 이미지 빌드
```bash
docker build -t ootd-backend .
```

### Docker 컨테이너 실행
```bash
docker run -p 5000:5000 \
  -e WEATHER_API_KEY_ENCODE=your_key \
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

### 단위 테스트 실행
```bash
# 전체 테스트 실행
pytest tests/

# 특정 테스트 파일 실행
pytest tests/test_api.py
pytest tests\test_api.py -v

# 커버리지 포함 테스트
pytest --cov=app tests/
```

### API 테스트
```bash
# 헬스 체크
curl http://localhost:5000/api/health

# 추천 API 테스트 (이미지 없이)
curl -X POST http://localhost:5000/api/recommend \
  -H "Content-Type: application/json" \
  -d '{"location": "서울", "latitude": 37.5665, "longitude": 126.9780, "user_request": "캐주얼한 스타일"}'
```

## 👨‍💻 개발자 가이드

### 새로운 API 엔드포인트 추가

1. **스키마 정의** (`schemas.py`):
```python
new_endpoint_schema = {
    "summary": "새로운 엔드포인트",
    "parameters": [...],
    "responses": {...}
}
```

2. **서비스 로직 구현** (`services.py`):
```python
def new_service_function():
    # 비즈니스 로직 구현
    return result
```

3. **라우트 등록** (`routes.py`):
```python
@app.route("/api/new-endpoint", methods=["POST"])
@swag_from(new_endpoint_schema)
def new_endpoint():
    # 라우트 로직
    return jsonify(result)
```

### AI 모델 추가

1. **모델 래퍼 클래스 생성** (`app/ai/new_model.py`):
```python
class NewModel:
    def __init__(self, model_path):
        # 모델 초기화
        pass
    
    def predict(self, input_data):
        # 추론 로직
        return predictions
```

2. **서비스에 모델 통합** (`services.py`):
```python
def initialize_ai_models(app):
    global new_model
    new_model = NewModel(model_path)
```

### 로깅 설정

로그 레벨 및 포맷은 `app/__init__.py`에서 설정:
```python
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s"
)
```

### 성능 모니터링

- **메모리 사용량**: AI 모델 로드 시 메모리 사용량 모니터링
- **응답 시간**: 각 API 엔드포인트의 응답 시간 측정
- **GPU 활용률**: CUDA 사용 시 GPU 메모리 및 활용률 확인

## 🐛 문제 해결

### 자주 발생하는 문제들

1. **AI 모델이 로드되지 않는 경우**:
   - 모델 파일 경로 확인
   - GPU 메모리 부족 여부 확인
   - CUDA 버전 호환성 확인

2. **날씨 API 오류**:
   - API 키 유효성 확인
   - 네트워크 연결 상태 확인
   - API 요청 제한 확인

3. **메모리 부족 오류**:
   - 이미지 배치 크기 조정
   - 모델 가중치 정밀도 조정 (FP16 사용)
   - 메모리 캐시 클리어

### 로그 확인

```bash
# 실시간 로그 모니터링
tail -f logs/app.log

# 에러 로그 필터링
grep "ERROR" logs/app.log

# 성능 관련 로그
grep "performance" logs/app.log
```

## 📞 지원 및 문의

- **이슈 리포트**: GitHub Issues 사용
- **개발 문의**: 프로젝트 팀에 연락
- **API 문서**: `/apidocs` 엔드포인트에서 실시간 문서 확인

---

**버전**: 1.0.0  
**마지막 업데이트**: 2024-01-01  
**라이선스**: MIT License
