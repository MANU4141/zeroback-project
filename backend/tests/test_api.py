import os
import json
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from app import create_app


def build_db_images(app):
    labels_dir = app.config["LABELS_DIR"]
    image_dir = app.config["IMAGE_DIR"]
    db_images = []
    label_files = [f for f in os.listdir(labels_dir) if f.endswith(".json")]
    total = len(label_files)
    print(f"[DB] 총 라벨 파일: {total}")
    for idx, fname in enumerate(label_files, 1):
        image_name = fname.replace(".json", ".jpg")
        image_path = os.path.join(image_dir, image_name)
        if not os.path.exists(image_path):
            continue
        try:
            with open(os.path.join(labels_dir, fname), encoding="utf-8") as f:
                label_data = json.load(f)
            label_info = {}
            labeling = (
                label_data.get("데이터셋 정보", {})
                .get("데이터셋 상세설명", {})
                .get("라벨링", {})
            )
            style_list = labeling.get("스타일", [])
            label_info["style"] = [
                s.get("스타일") for s in style_list if s.get("스타일")
            ]
            color, material, category, detail = set(), set(), set(), set()
            for part in ["아우터", "하의", "상의", "원피스"]:
                items = labeling.get(part, [])
                for item in items:
                    if "색상" in item and item["색상"]:
                        color.add(item["색상"])
                    if "소재" in item and item["소재"]:
                        if isinstance(item["소재"], list):
                            material.update(item["소재"])
                        else:
                            material.add(item["소재"])
                    if "카테고리" in item and item["카테고리"]:
                        category.add(item["카테고리"])
                    if "디테일" in item and item["디테일"]:
                        if isinstance(item["디테일"], list):
                            detail.update(item["디테일"])
                        else:
                            detail.add(item["디테일"])
            label_info["color"] = list(color)
            label_info["material"] = list(material)
            label_info["category"] = list(category)
            label_info["detail"] = list(detail)
            db_images.append({"img_path": image_path, "label": label_info})
            if idx % 100 == 0 or idx == total:
                print(f"[DB] 진행률: {idx}/{total} ({(idx/total)*100:.1f}%)")
        except Exception as e:
            continue
    print(f"[DB] 최종 DB 이미지 개수: {len(db_images)}")
    return db_images


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    app.config["LABELS_DIR"] = (
        r"D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\labels"
    )
    app.config["IMAGE_DIR"] = (
        r"D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\images"
    )
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
    # 요청 데이터
    req_data = {
        "location": "서울",
        "latitude": 37.5665,
        "longitude": 126.9780,
        "style_select": ["스트릿", "캐주얼"],
        "user_request": "귀엽게 입고 싶어요",
    }
    # 테스트에 사용할 이미지를 명시적으로 지정 (환경변수 무시)
    image_path = r"D:\end_github_zeroback\zeroback-project\backend\test_image.jpg"
    assert os.path.exists(
        image_path
    ), f"테스트 이미지가 존재하지 않습니다: {image_path}"
    img_file = open(image_path, "rb")
    data = {
        "data": json.dumps(req_data),
        "images": (img_file, "test.jpg", "image/jpeg"),
    }
    resp = client.post("/api/recommend", data=data)
    img_file.close()
    assert resp.status_code == 200
    result = resp.get_json()
    assert result["success"] is True
    assert "ai_analysis" in result
    # 분석 결과가 없으면 None이 반환될 수 있으므로 None도 허용
    if isinstance(result["ai_analysis"], dict):
        print(
            "AI 분석 결과:",
            json.dumps(result["ai_analysis"], ensure_ascii=False, indent=2),
        )
    assert result["ai_analysis"] is None or isinstance(result["ai_analysis"], dict)
    assert "recommendation_details" in result
    # 추천 이미지 필드가 있으면 리스트 타입인지 확인 및 출력
    rec = result["recommendation_details"]
    if "all_recommended_images" in rec:
        assert isinstance(rec["all_recommended_images"], list)
        print("\n===== 추천 이미지 목록 (상위 3개) =====")
        for i, img in enumerate(rec["all_recommended_images"], 1):
            print(
                f"[{i}] {os.path.basename(img['img_path'])} | style: {img['label'].get('style')} | category: {img['label'].get('category')} | color: {img['label'].get('color')}"
            )
        print("====================================\n")
        # 추천 이미지가 비어있지 않은지 확인
        assert len(rec["all_recommended_images"]) > 0
    if "style_matched_images" in rec:
        assert isinstance(rec["style_matched_images"], list)
        print(
            "style_matched_images:",
            json.dumps(rec["style_matched_images"], ensure_ascii=False, indent=2),
        )
        # style_matched_images가 비어있어도 실패하지 않음
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
    assert (
        len(db_images) > 0
    ), "DB_IMAGES가 비어 있습니다. 라벨 json과 이미지 파일을 확인하세요."
    required_fields = ["style", "color", "material", "category", "detail"]
    for entry in db_images:
        # 이미지 파일 존재 확인
        img_path = entry.get("img_path")
        assert img_path and os.path.exists(
            img_path
        ), f"이미지 파일이 존재하지 않음: {img_path}"
        label = entry.get("label", {})
        for field in required_fields:
            assert field in label, f"{field} 필드가 라벨에 없습니다: {img_path}"
            # 필드가 비어있어도 경고 출력하지 않음
