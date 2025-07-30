## 폴더 구조

# OOTD-AI Project

### _Ai가 추천하는 오늘의 스타일_

2025년 가천x세종 연합 학술제 팀 제로백의 산출물인 OOTD-AI는 예시 스타일 이미지/사용자의 위치에 따른 날씨/사용자의 요구사항을 바탕으로 의류를 추천하는 서비스입니다.

## Features

- 웹 인터페이스
- 위치 기반 날씨 정보 수집
- 버튼과 텍스트 통한 사용자 스타일 입력
- 이미지 업로드를 통한 예시 스타일 확인
- 입력받은 정보를 종합하여 이미지 추천
- LLM Api 연동 통한 스타일 추천 문구 생성

## Tech

아래와 같은 기술을 사용하고 있습니다.

- Flask - 내부 api 및 외부 api 서빙 백엔드
- YOLO - 이미지 추천

## Installation

## 1. 사전 준비

- `config/config.py`에서 사용할 YOLO 모델(`.pt`) 경로를 `MODEL_PATHS["yolo"]`에 지정하세요.
- `CLASS_MAPPINGS["category"]`의 클래스 수가 실제 모델의 클래스 수와 반드시 일치해야 합니다.
- 분석할 이미지를 `Algorithm/AI/images/test.jpg`로 준비하세요.

---

## 2. 실행 방법

1. **필수 패키지 설치**
    pip install torch ultralytics opencv-python pillow numpy

2. **이미지 분석 및 추천 실행**
    cd Algorithm/AI
    python yolo_test.py

3. **결과 확인**
    - 터미널에 탐지 결과, 속성, 추천 결과가 출력됩니다.
    - `images/pred_result.jpg`에 바운딩박스와 속성 라벨이 시각화된 이미지가 저장됩니다.

---

## 3. 주요 코드 설명

- **yolo_test.py**  
  YOLO 모델을 로드하여 이미지를 분석하고, 탐지된 객체의 속성을 예측합니다.  
  예측된 속성 정보를 바탕으로 추천 결과를 생성하고, 결과 이미지를 저장합니다.

- **config/config.py**  
  모델 경로와 클래스 매핑 정보를 관리합니다.  
  클래스 리스트는 실제 모델의 학습 클래스와 반드시 일치해야 합니다.

- **AI/yolo_multitask.py**  
  YOLO 모델을 여러 속성(task) 예측이 가능하도록 래핑합니다.

- **recommender/final_recommender.py**  
  예측된 속성, 날씨, 사용자 프롬프트 등을 바탕으로 추천 결과를 생성합니다.

---

## 4. 주의사항

- 모델(`YOLOv11_large.pt` 등)의 클래스 수와 `CLASS_MAPPINGS`의 리스트 길이가 다르면 예측이 제대로 동작하지 않습니다.
- 이미지 파일 경로, 모델 경로가 올바른지 확인하세요.
- 한글 폰트가 시스템에 없을 경우 기본 폰트로 대체됩니다.

---

## 5. 예시 실행 결과

- 터미널에 탐지된 객체 수, 각 객체의 바운딩박스/속성/추천 결과가 출력됩니다.
- `images/pred_result.jpg`에서 시각화된 결과 이미지를 확인할 수 있습니다.

---

**오류 발생 시**
- 모델 클래스 수와 `CLASS_MAPPINGS` 불일치
- 이미지/모델 경로 오류
- 한글 폰트 미설치
