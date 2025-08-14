# Backend 파일 구조 및 기능 설명

## 📁 Root Files
- `config.py` : 환경 설정 및 모델 경로, API 키, 클래스 매핑 정의
- `run.py` : Flask 애플리케이션 진입점 및 서버 실행
- `requirements.txt` : Python 의존성 패키지 목록
- `Dockerfile` : Docker 이미지 빌드 설정
- `swagger.yaml` : API 문서 스키마 정의

## 📁 app/
- `__init__.py` : Flask 앱 팩토리 및 초기화 설정
- `routes.py` : API 엔드포인트 라우트 정의
- `services.py` : AI 모델 초기화 및 비즈니스 로직
- `schemas.py` : API 스키마 및 Swagger 문서 정의
- `utils.py` : 이미지 처리 및 유틸리티 함수
- `weather.py` : 기상청 날씨 API 클라이언트

## 📁 app/ai/
- `__init__.py` : AI 모듈 초기화
- `yolo_multitask.py` : YOLO 객체 탐지 모델 래퍼
- `resnet_multitask.py` : ResNet 다중 속성 분류 모델 래퍼

## 📁 app/llm/
- `__init__.py` : LLM 모듈 초기화
- `gemini_prompt_utils.py` : Google Gemini API 통합 및 프롬프트 처리

## 📁 app/recommender/
- `__init__.py` : 추천 시스템 모듈 초기화
- `db_similarity.py` : 이미지 유사도 계산 로직
- `final_recommender.py` : 최종 의상 추천 알고리즘

## 📁 DATA/
- `images/` : 의류 이미지 데이터베이스 (1000+ 이미지)
- `labels/` : 이미지 라벨링 데이터

## 📁 models/
- `ResNet50_45.pth` : 학습된 ResNet50 다중 속성 분류 모델
- `YOLOv11_large.pt` : 학습된 YOLO 객체 탐지 모델

## 📁 tests/
- `__init__.py` : 테스트 모듈 초기화
- `test_api.py` : API 엔드포인트 단위 테스트
