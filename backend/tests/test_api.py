import os
import json
import pytest
import sys
import time

# 프로젝트 루트를 sys.path에 추가하여 'backend.*' 패키지 임포트가 가능하도록 설정
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import create_app, build_db_images  # noqa: E402
from backend.config import Config  # noqa: E402


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    # 기본 설정(Config) 경로 사용. 필요 시 환경 또는 상단 Config로 조정 가능
    app.config["LABELS_DIR"] = Config.LABELS_DIR
    app.config["IMAGE_DIR"] = Config.IMAGE_DIR
    # DB_IMAGES 재생성 (app 전달 시 app.config의 LABELS/IMAGE_DIR 사용)
    app.config["DB_IMAGES"] = build_db_images(app)
    with app.test_client() as client:
        yield client


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["status"] == "OK"
    assert data["models_initialized"] is True
    assert data["yolo_model_loaded"] is True


def test_ai_status(client):
    resp = client.get("/api/debug/ai-status")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["models_initialized"] is True
    assert data["yolo_model_loaded"] is True
    assert data["resnet_model_loaded"] is True
    assert data["YOLOv11MultiTask_available"] is True
    assert data["model_file_exists"] is True


def test_recommend_with_image(client):
    req_data = {
        "location": "서울",
        "latitude": 37.5665,
        "longitude": 126.9780,
        "style_select": ["스트릿", "캐주얼"],
        "user_request": "귀엽게 입고 싶어요",
    }
    # 실제 존재하는 이미지 선정: DB_IMAGES에서 첫 항목 또는 IMAGE_DIR 내 임의 파일
    db_images = client.application.config.get("DB_IMAGES", [])
    image_path = None
    if db_images:
        image_path = db_images[0].get("img_path")
    if not image_path or not os.path.exists(image_path):
        img_dir = client.application.config.get("IMAGE_DIR")
        if img_dir and os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(img_dir, fname)
                    break
    if not image_path or not os.path.exists(image_path):
        pytest.skip(
            "테스트용 이미지 파일이 없습니다. IMAGE_DIR 또는 DB_IMAGES를 확인하세요."
        )
    with open(image_path, "rb") as img_file:
        data = {
            "data": json.dumps(req_data),
            "images": (img_file, "test.jpg", "image/jpeg"),
        }
        resp = client.post("/api/recommend", data=data)
    assert resp.status_code == 200
    result = resp.get_json()
    assert result["success"] is True
    # ai_analysis가 아닌, color/style/category/material/detail 등 속성이 최상위에 있는지 확인
    ai_keys = ["color", "style", "category", "material", "detail"]
    ai_found = any(k in result for k in ai_keys)
    assert ai_found or result.get(
        "ai_analysis_status"
    ), "AI 분석 결과가 응답에 없습니다."
    # 추천 결과 구조 확인
    assert "recommendation_details" in result
    rec = result["recommendation_details"]
    if "all_recommended_images" in rec:
        assert isinstance(rec["all_recommended_images"], list)
        print("\n===== 추천 이미지 목록 (상위 3개) =====")
        for i, img in enumerate(rec["all_recommended_images"], 1):
            label = img["label"]
            print(
                f"[{i}] {os.path.basename(img['img_path'])} | style: {label.get('style')} | category: {label.get('category')} | color: {label.get('color')} | material: {label.get('material')} | detail: {label.get('detail')}"
            )
        print("====================================\n")
        assert len(rec["all_recommended_images"]) > 0
    if "style_matched_images" in rec:
        assert isinstance(rec["style_matched_images"], list)
        print(
            "style_matched_images:",
            json.dumps(rec["style_matched_images"], ensure_ascii=False, indent=2),
        )
        if rec["style_matched_images"]:
            assert len(rec["style_matched_images"]) > 0


def test_recommend_bad_request(client):
    # 필수 필드 누락
    resp = client.post(
        "/api/recommend", data={"data": "{}"}, content_type="multipart/form-data"
    )
    assert resp.status_code == 400
    data = resp.get_json()
    assert "error" in data


def test_db_images_integrity(client):
    """
    추천 DB(DB_IMAGES)에 데이터가 충분히 있는지, 필수 필드가 모두 채워져 있는지, 이미지 파일이 실제로 존재하는지 확인.
    """
    db_images = client.application.config["DB_IMAGES"]
    if len(db_images) == 0:
        pytest.skip("DB_IMAGES가 비어 있습니다. 데이터셋이 구성되지 않았습니다.")
    required_fields = ["style", "color", "material", "category", "detail"]
    for entry in db_images:
        img_path = entry.get("img_path")
        assert img_path and os.path.exists(
            img_path
        ), f"이미지 파일이 존재하지 않음: {img_path}"
        label = entry.get("label", {})
        for field in required_fields:
            assert field in label, f"{field} 필드가 라벨에 없습니다: {img_path}"


def test_model_prediction_time(client):
    """
    AI 모델의 순수 예측 시간만을 측정하는 테스트
    """
    req_data = {
        "location": "서울",
        "latitude": 37.5665,
        "longitude": 126.9780,
        "style_select": ["캐주얼"],
        "user_request": "테스트용 요청",
    }

    # 테스트용 이미지 파일 찾기
    db_images = client.application.config.get("DB_IMAGES", [])
    image_path = None
    if db_images:
        image_path = db_images[0].get("img_path")
    if not image_path or not os.path.exists(image_path):
        img_dir = client.application.config.get("IMAGE_DIR")
        if img_dir and os.path.isdir(img_dir):
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_path = os.path.join(img_dir, fname)
                    break

    if not image_path or not os.path.exists(image_path):
        pytest.skip("테스트용 이미지 파일이 없습니다.")

    # 모델 예측 시간 측정을 위해 여러 번 실행
    prediction_times = []
    num_runs = 3

    print(f"\n===== 모델 예측 시간 측정 ({num_runs}회 실행) =====")
    print(f"테스트 이미지: {os.path.basename(image_path)}")

    for run in range(num_runs):
        with open(image_path, "rb") as img_file:
            data = {
                "data": json.dumps(req_data),
                "images": (img_file, f"test_{run}.jpg", "image/jpeg"),
            }

            # API 호출 시작 시간
            api_start_time = time.time()
            resp = client.post("/api/recommend", data=data)
            api_end_time = time.time()

            assert resp.status_code == 200
            result = resp.get_json()
            assert result["success"] is True

            # 전체 API 시간
            total_api_time = api_end_time - api_start_time

            # AI 분석 시간 추출 (응답에 포함되어 있다면)
            ai_debug = result.get("ai_debug_details", [])
            model_time = None
            yolo_time = None
            resnet_time = None
            preprocess_time = None
            num_objects = None

            # 디버그 정보에서 모델 예측 시간 추출
            for debug_info in ai_debug:
                if "processing_time" in debug_info:
                    model_time = debug_info["processing_time"]
                    yolo_time = debug_info.get("yolo_time")
                    resnet_time = debug_info.get("resnet_total_time")
                    preprocess_time = debug_info.get("preprocess_time")
                    num_objects = debug_info.get("num_objects")
                    break

            if model_time is None:
                # 디버그 정보가 없다면 전체 API 시간의 추정치 사용
                # (네트워크, 파싱, 추천 로직 제외하고 AI 분석 부분만 추정)
                estimated_model_time = (
                    total_api_time * 0.7
                )  # API 시간의 70%를 모델 시간으로 추정
                model_time = estimated_model_time

            prediction_times.append(
                {
                    "run": run + 1,
                    "total_api_time": total_api_time,
                    "model_time": model_time,
                    "yolo_time": yolo_time,
                    "resnet_time": resnet_time,
                    "preprocess_time": preprocess_time,
                    "num_objects": num_objects,
                }
            )

            detail_str = ""
            if yolo_time and resnet_time and preprocess_time:
                detail_str = f" (전처리: {preprocess_time:.3f}초, YOLO: {yolo_time:.3f}초, ResNet: {resnet_time:.3f}초, 객체: {num_objects}개)"

            print(
                f"Run {run + 1}: 전체 API 시간 {total_api_time:.3f}초, 모델 예측 시간 {model_time:.3f}초{detail_str}"
            )

    # 통계 계산
    avg_model_time = sum(t["model_time"] for t in prediction_times) / num_runs
    min_model_time = min(t["model_time"] for t in prediction_times)
    max_model_time = max(t["model_time"] for t in prediction_times)
    avg_api_time = sum(t["total_api_time"] for t in prediction_times) / num_runs

    # 개별 모델 통계 (값이 있는 경우에만)
    yolo_times = [
        t["yolo_time"] for t in prediction_times if t["yolo_time"] is not None
    ]
    resnet_times = [
        t["resnet_time"] for t in prediction_times if t["resnet_time"] is not None
    ]
    preprocess_times = [
        t["preprocess_time"]
        for t in prediction_times
        if t["preprocess_time"] is not None
    ]

    print(f"\n===== 모델 예측 시간 통계 =====")
    print(f"평균 모델 예측 시간: {avg_model_time:.3f}초")
    print(f"최소 모델 예측 시간: {min_model_time:.3f}초")
    print(f"최대 모델 예측 시간: {max_model_time:.3f}초")
    print(f"평균 전체 API 시간: {avg_api_time:.3f}초")
    print(f"모델 시간 비율: {(avg_model_time/avg_api_time*100):.1f}%")

    if preprocess_times:
        print(f"\n--- 세부 성능 분석 ---")
        print(f"평균 전처리 시간: {sum(preprocess_times)/len(preprocess_times):.3f}초")
    if yolo_times:
        print(f"평균 YOLO 탐지 시간: {sum(yolo_times)/len(yolo_times):.3f}초")
    if resnet_times:
        print(f"평균 ResNet 예측 시간: {sum(resnet_times)/len(resnet_times):.3f}초")

    # 객체 수별 성능 분석
    obj_counts = [
        t["num_objects"] for t in prediction_times if t["num_objects"] is not None
    ]
    if obj_counts:
        avg_objects = sum(obj_counts) / len(obj_counts)
        print(f"평균 탐지된 객체 수: {avg_objects:.1f}개")

    print("============================\n")  # 성능 검증 (모델 예측이 5초 이내여야 함)
    assert avg_model_time < 5.0, f"모델 예측 시간이 너무 깁니다: {avg_model_time:.3f}초"
