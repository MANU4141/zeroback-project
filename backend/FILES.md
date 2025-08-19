
# Backend 파일 구조 및 기능 설명 (2025.08 기준)

## 📁 루트 파일
- `config.py` : 환경설정, 경로, API 키, 클래스 매핑 등 백엔드 전체 설정
- `run.py` : Flask 앱 실행 진입점
- `requirements.txt` : Python 의존성 패키지 목록
- `Dockerfile` : 백엔드용 Docker 빌드 설정
- `swagger.yaml` : API 문서 스키마
- `FILES.md` : 이 파일, 백엔드 파일 구조 설명

## 📁 app/
- `__init__.py` : Flask 앱 팩토리 및 초기화
- `error_codes.py` : 에러 코드 및 에러 응답 생성
- `monitoring.py` : 성능 모니터링 유틸리티
- `schemas.py` : API 스키마 및 Swagger 문서
- `services.py` : AI 모델 관리, 날씨/추천/분석 등 주요 서비스 로직
- `utils.py` : 이미지/경로/DB 유틸리티 함수
- `weather.py` : 기상청 날씨 API 클라이언트

### 📁 app/ai/
- `__init__.py` : AI 모듈 초기화
- `resnet_multi_classification.py` : ResNet50 다중 속성 분류 래퍼
- `yolo_classification.py` : YOLOv11 객체 탐지 래퍼

### 📁 app/llm/
- `__init__.py` : LLM 모듈 초기화
- `gemini_prompt_utils.py` : Google Gemini API 프롬프트/분석 유틸

### 📁 app/recommender/
- `__init__.py` : 추천 시스템 초기화
- `db_similarity.py` : 이미지 유사도 계산
- `final_recommender.py` : 최종 의상 추천 알고리즘
- `style_mappings.py` : 스타일/속성 매핑

### 📁 app/routes/
- `__init__.py` : 라우트 등록
- `debug.py` : 디버그/테스트용 라우트
- `health.py` : 헬스체크 및 상태 라우트
- `recommendation.py` : 추천 API 라우트
- `utility.py` : 유틸리티 API 라우트

### 📁 config/
- `class_mappings.json` : 카테고리/클래스 매핑

## 📁 DATA/
- `images/` : 의류 이미지 데이터베이스 (3000+ 이미지)
- `labels/` : 이미지 라벨링 데이터
- `labels_json/` : 라벨 JSON 데이터

## 📁 models/
- `ResNet50.pth` : 학습된 ResNet50 모델
- `YOLOv11_large.pt` : 학습된 YOLOv11 모델
- `YOLOv11_large_last_100.pt` : 최근 학습된 YOLOv11 모델

## 📁 tests/
- `__init__.py` : 테스트 모듈 초기화
- `prototype_test.py` : 백엔드 통합 프로토타입 테스트 (AI/날씨/추천)
- `test_api.py` : API 엔드포인트 단위 테스트

## 기타
- `Freesentation-2ExtraLight.ttf` : 폰트 파일
- `EX.md` : 예시/샘플 문서


