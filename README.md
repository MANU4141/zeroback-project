## 백엔드 테스트 코드 실행법 (test_api.py)

`backend/tests/test_api.py`는 백엔드 서버의 주요 API와 데이터베이스, 추천 기능을 통합적으로 테스트하는 코드입니다.

### 실행 방법

1. 의존성 설치 및 환경 변수 설정(상단 "실행 방법 및 필요 사항" 참고)
2. 테스트용 이미지(`backend/test_image.jpg`)와 라벨/이미지 데이터가 실제 경로에 존재해야 합니다.
3. 아래 명령어로 테스트 실행:

```powershell
cd backend
pytest tests/test_api.py
```

### 주의사항
- `test_api.py` 내부에서 라벨/이미지 경로가 하드코딩되어 있으므로, 실제 데이터가 해당 경로에 있어야 테스트가 정상 동작합니다.
  - 예시: `D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\labels` 등
- 테스트 중 모델 로딩 및 DB 이미지 빌드에 시간이 다소 소요될 수 있습니다.
- 테스트 결과는 콘솔에 출력되며, 일부 테스트는 추천 이미지 목록 등도 함께 출력합니다.

# KHJ_README (Backend)

## 프로젝트 개요
- Flask 기반 AI/추천/날씨 API 서버
- Swagger 문서: `/apidocs/`
- 주요 엔드포인트:
  - `/api/health` : 서버/모델 상태 확인
  - `/api/recommend` : 옷 추천(이미지+텍스트+날씨)

## 환경 변수
- `LABELS_DIR` : 라벨 json 폴더 경로
- `IMAGE_DIR` : 이미지 폴더 경로
- `DOC_USER`, `DOC_PASS` : Swagger 인증(옵션)
- `WEATHER_API_KEY_DECODE` : 기상청 API 키

## 주요 파라미터/입력
- `/api/recommend` (POST, multipart/form-data)
  - `data`: JSON (location, latitude, longitude, style_select, user_request)
  - `images` or `image`: 이미지 파일

예시 data:
```json
{
  "location": "Seoul",
  "latitude": 37.5665,
  "longitude": 126.9780,
  "style_select": ["캐주얼", "스트릿"],
  "user_request": "데이트룩"
}
```

## 참고
- Swagger 문서에서 모든 파라미터/응답 예시 확인 가능
- 에러/로그는 컨테이너 stdout 또는 logs 폴더 참고

---


## 실행 방법 및 필요 사항

### 1. 필수 환경
- Python 3.8 이상
- pip (Python 패키지 매니저)
  - (선택) Docker, docker-compose (컨테이너 실행 시)

```powershell
# 백엔드 의존성 설치
cd backend
pip install -r requirements.txt

# (필요시) 알고리즘/AI 의존성 설치
cd ../Algorithm
pip install -r requirements.txt
```

### 3. 환경 변수 설정
- `LABELS_DIR`, `IMAGE_DIR` 등 경로 환경변수 필요 (예: 라벨/이미지 데이터셋 경로)
- 테스트 실행 시, `test_api.py`에서 경로가 하드코딩되어 있으니 실제 데이터가 해당 위치에 있어야 함

예시:
```
LABELS_DIR=D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\labels
IMAGE_DIR=D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\images
```

### 4. 서버 실행
```powershell
# 백엔드 서버 실행
cd backend
python app.py
```
- Swagger 문서: http://localhost:5000/apidocs/

### 5. 테스트 실행
```powershell
cd backend
pytest tests/test_api.py
```
- 테스트 이미지(`test_image.jpg`)와 라벨/이미지 데이터가 실제 경로에 있어야 정상 동작

### 6. Docker로 실행 (선택)
```powershell
docker-compose up --build
```
- `docker-compose.yml` 참고

---

## 백엔드 주요 파일/로직 요약 (최소설명)

- `backend/app.py` : Flask 서버 진입점, 모든 API 라우팅, 모델/DB/날씨 초기화, 요청 파싱/응답
- `backend/ai/resnet_multitask.py` : ResNet50 기반 옷 속성 추론(색상, 스타일 등), 이미지→속성 벡터화
- `backend/ai/yolo_multitask.py` : YOLOv11 기반 옷/객체 탐지, 이미지에서 옷 부분 crop
- `backend/recommender/final_recommender.py` : 날씨/유저프롬프트/AI분석 결과를 종합해 top-N 옷 추천(가중치 합산)
- `backend/weather_api.py` : 기상청 API 연동, 위경도→격자 변환, 날씨 데이터 파싱
- `backend/config/config.py` : 모델 경로, 클래스/속성 리스트 등 모든 설정값 중앙 관리
- `backend/llm/gemini_prompt_utils.py` : LLM(Gemini) 프롬프트 분석, 유저 요청에서 키워드 추출
- `backend/tests/` : 백엔드 단위테스트

### 전체 추천 알고리즘 흐름
1. 이미지 업로드 → YOLO로 옷 crop → ResNet으로 속성 추론
2. 유저 프롬프트/스타일/날씨 정보 파싱
3. DB 이미지와 속성/스타일/날씨/프롬프트 가중치 기반 유사도 계산
4. 최종 top-N 추천 결과 반환

---
