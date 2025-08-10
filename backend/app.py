# -*- coding: utf-8 -*-
from flask import Flask, request, jsonify, current_app
from flask_cors import CORS
from flasgger import swag_from
from datetime import datetime
import json
import yaml
import os
import logging
from functools import lru_cache

# AI 모듈 import
import cv2
import numpy as np
from PIL import Image
import io

# Swagger 설정 import
from swagger_config import init_swagger, swagger_template
from api_schemas import (
    recommend_schema,
    health_check_schema,
    make_swagger_schema,
    debug_ai_status_schema,
    debug_weather_test_schema,
)

# AI 모델 클래스와 기타 유틸리티 import
try:
    from ai.yolo_multitask import YOLOv11MultiTask
    from ai.resnet_multitask import FashionAttributePredictor
    from recommender.final_recommender import final_recommendation
    from config.config import CLASS_MAPPINGS, MODEL_PATHS
    from weather.recommend_by_weather import recommend_by_weather
    from llm.gemini_prompt_utils import analyze_user_prompt
    from weather_api import KoreaWeatherAPI
except ImportError as e:
    print(f"AI 모듈 import 실패: {e}")
    YOLOv11MultiTask = None
    FashionAttributePredictor = None
    KoreaWeatherAPI = None

yolo_model = None  # YOLOv11MultiTask 인스턴스
resnet_model = None  # FashionAttributePredictor 인스턴스
weather_api = None
_models_initialized = False

SWAGGER_YAML_PATH = os.getenv(
    "SWAGGER_YAML_PATH",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "swagger.yaml"),
)


def create_app():
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})
    app.config["SWAGGER"] = {
        "title": "API Docs",
        "uiversion": 3,
        "specs_route": "/apidocs/",
        "hide_top_bar": True,
    }

    @lru_cache(maxsize=1)
    def build_db_images():
        labels_dir = app.config["LABELS_DIR"]
        image_dir = app.config["IMAGE_DIR"]
        db_images = []
        total = 0
        for fname in os.listdir(labels_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(labels_dir, fname), encoding="utf-8") as f:
                    label_data = json.load(f)
                image_name = label_data.get("이미지 정보", {}).get("이미지 파일명")
                if not image_name:
                    continue
                image_path = os.path.join(image_dir, image_name)
                if not os.path.exists(image_path):
                    continue
                label_info = {}
                style_list = (
                    label_data.get("데이터셋 정보", {})
                    .get("라벨링", {})
                    .get("스타일", [])
                )
                label_info["style"] = [
                    s.get("스타일") for s in style_list if s.get("스타일")
                ]
                color_list = (
                    label_data.get("데이터셋 정보", {})
                    .get("라벨링", {})
                    .get("색상", [])
                )
                label_info["color"] = [
                    c.get("색상") for c in color_list if c.get("색상")
                ]
                material_list = (
                    label_data.get("데이터셋 정보", {})
                    .get("라벨링", {})
                    .get("소재", [])
                )
                label_info["material"] = [
                    m.get("소재") for m in material_list if m.get("소재")
                ]
                category_list = (
                    label_data.get("데이터셋 정보", {})
                    .get("라벨링", {})
                    .get("카테고리", [])
                )
                label_info["category"] = [
                    c.get("카테고리") for c in category_list if c.get("카테고리")
                ]
                detail_list = (
                    label_data.get("데이터셋 정보", {})
                    .get("라벨링", {})
                    .get("디테일", [])
                )
                label_info["detail"] = [
                    d.get("디테일") for d in detail_list if d.get("디테일")
                ]
                db_images.append({"img_path": image_path, "label": label_info})
                total += 1
                if total % 100 == 0:
                    logging.info(f"[DB] 진행률: {total}개")
            except Exception as e:
                logging.warning(f"[DB] 라벨 파싱 실패: {fname}, {e}")
                continue
        logging.info(f"[DB] 최종 DB 이미지 개수: {len(db_images)}")
        return db_images

    app.config["DB_IMAGES"] = build_db_images()

    # Swagger UI Basic Auth (optional)
    doc_user = os.getenv("DOC_USER")
    doc_pass = os.getenv("DOC_PASS")
    if doc_user and doc_pass:
        from flask_httpauth import HTTPBasicAuth

        auth = HTTPBasicAuth()
        users = {doc_user: doc_pass}

        @auth.get_password
        def get_pw(username):
            return users.get(username)

        @app.before_request
        def restrict_swagger():
            if request.path.startswith("/apidocs"):
                return auth.login_required(lambda: None)()

    init_swagger(app)

    # 날씨 API 세팅
    app.config["weather_api"] = KoreaWeatherAPI()
    global weather_api
    weather_api = app.config["weather_api"]

    # AI 모델 초기화
    success = initialize_ai_models(app)
    if not success:
        app.logger.error("AI 모델 초기화에 실패했습니다.")
    global _models_initialized
    _models_initialized = success

    # --- ROUTES ---
    @app.route("/api/makeswagger", methods=["GET"])
    @swag_from(make_swagger_schema)
    def make_swagger():
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            yaml_file_path = os.path.join(current_dir, "swagger.yaml")
            swagger_spec = swagger_template.copy()
            swagger_spec["paths"] = {
                "/api/recommend": {"post": recommend_schema},
                "/api/health": {"get": health_check_schema},
                "/api/makeswagger": {"get": make_swagger_schema},
                "/api/debug/ai-status": {"get": debug_ai_status_schema},
                "/api/debug/weather-test": {"get": debug_weather_test_schema},
            }
            with open(yaml_file_path, "w", encoding="utf-8") as f:
                yaml.dump(
                    swagger_spec,
                    f,
                    allow_unicode=True,
                    default_flow_style=False,
                    sort_keys=False,
                )
            return (
                jsonify(
                    {
                        "message": "Swagger YAML file successfully created",
                        "file_path": yaml_file_path,
                        "timestamp": datetime.now().isoformat(),
                        "file_size": f"{os.path.getsize(yaml_file_path)} bytes",
                    }
                ),
                200,
            )
        except Exception as e:
            return (
                jsonify(
                    {
                        "error": f"Swagger file creation failed: {str(e)}",
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                500,
            )

    @app.route("/api/health", methods=["GET"])
    @swag_from(health_check_schema)
    def health_check():
        return (
            jsonify(
                {
                    "status": "OK",
                    "timestamp": datetime.now().isoformat(),
                    "models_initialized": _models_initialized,
                    "yolo_model_loaded": bool(yolo_model),
                    "resnet_model_loaded": bool(resnet_model),
                }
            ),
            200,
        )

    @app.route("/api/recommend", methods=["POST"])
    @swag_from(recommend_schema)
    def recommend():
        # Content-Type 검사
        content_type = request.content_type
        if not (content_type and content_type.startswith("multipart/form-data")):
            return (
                jsonify({"error": "Unsupported data format, use multipart/form-data"}),
                400,
            )

        # JSON 데이터 파싱
        json_data = request.form.get("data")
        if not json_data:
            return jsonify({"error": "No data received"}), 400
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError:
            return jsonify({"error": "Invalid JSON format"}), 400

        location = data.get("location")
        latitude = data.get("latitude")
        longitude = data.get("longitude")
        style_select = data.get("style_select")
        user_request = data.get("user_request")

        # 위도/경도 형 변환
        try:
            if latitude is not None and not isinstance(latitude, float):
                latitude = float(latitude)
            if longitude is not None and not isinstance(longitude, float):
                longitude = float(longitude)
        except Exception:
            return jsonify({"error": "Latitude and longitude must be numbers."}), 400

        # 필수 필드 검사
        if not location:
            return jsonify({"error": "Location is required"}), 400
        if latitude is None or longitude is None:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        if not user_request or not user_request.strip():
            return jsonify({"error": "User request is required"}), 400

        # 이미지 파일 파싱
        uploaded_files = parse_images()
        current_app.logger.debug(f"Uploaded files count: {len(uploaded_files)}")

        ai_attributes = None
        ai_debug_details = []

        # AI 분석
        if uploaded_files and yolo_model and resnet_model:
            results_list = []
            for idx, uploaded_file in enumerate(uploaded_files):
                debug_info = {"id": idx, "filename": uploaded_file.filename}
                if not uploaded_file.filename:
                    debug_info["error"] = "No filename"
                    ai_debug_details.append(debug_info)
                    continue
                if uploaded_file.mimetype not in ["image/jpeg", "image/png"]:
                    debug_info["error"] = "지원되지 않는 이미지 형식"
                    ai_debug_details.append(debug_info)
                    continue
                try:
                    uploaded_file.seek(0)
                    image_bytes = uploaded_file.read()
                    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
                    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                    # 1) YOLO 탐지
                    try:
                        results = yolo_model.detect(image_np)
                        crops = yolo_model.extract_crops(image_np, results)
                        if not crops:
                            raise ValueError("객체 없음")
                        crop, bbox, conf, cls = crops[0]
                    except Exception as yolo_err:
                        debug_info["error"] = f"YOLO 탐지 실패: {yolo_err}"
                        ai_debug_details.append(debug_info)
                        results_list.append(
                            {
                                "id": idx,
                                "filename": uploaded_file.filename,
                                "attributes": None,
                                "confidence": 0.0,
                                "detected_class": -1,
                                "bbox": "No objects detected",
                            }
                        )
                        continue

                    # 2) ResNet 속성 예측
                    try:
                        image_attributes = resnet_model.predict_attributes(
                            crop, CLASS_MAPPINGS
                        )
                    except Exception as resnet_err:
                        debug_info["error"] = f"ResNet 예측 실패: {resnet_err}"
                        ai_debug_details.append(debug_info)
                        results_list.append(
                            {
                                "id": idx,
                                "filename": uploaded_file.filename,
                                "attributes": None,
                                "confidence": float(conf),
                                "detected_class": int(cls),
                                "bbox": (
                                    bbox.tolist()
                                    if hasattr(bbox, "tolist")
                                    else list(bbox)
                                ),
                            }
                        )
                        continue

                    # BBox 리스트 변환
                    if hasattr(bbox, "tolist"):
                        bbox_list = bbox.tolist()
                    else:
                        bbox_list = (
                            list(bbox)
                            if isinstance(bbox, (list, tuple))
                            else [float(bbox)]
                        )

                    results_list.append(
                        {
                            "id": idx,
                            "filename": uploaded_file.filename,
                            "attributes": image_attributes,
                            "confidence": float(conf),
                            "detected_class": int(cls),
                            "bbox": bbox_list,
                        }
                    )
                    debug_info["success"] = True
                    ai_debug_details.append(debug_info)

                except Exception as e:
                    debug_info["error"] = f"이미지 분석 오류: {e}"
                    ai_debug_details.append(debug_info)
                    results_list.append(
                        {
                            "id": idx,
                            "filename": uploaded_file.filename,
                            "error": str(e),
                            "confidence": 0.0,
                        }
                    )

            ai_attributes = combine_multiple_image_results(results_list)

        elif uploaded_files:
            ai_debug_details.append({"error": "AI 모델이 로드되지 않음"})
            current_app.logger.error("AI 모델이 로드되지 않았습니다 (YOLO/ResNet)")
        else:
            ai_debug_details.append({"error": "업로드된 이미지 없음"})
            current_app.logger.info("업로드된 이미지가 없습니다")

        # 스타일 선호도 결정
        style_preferences = style_select or []
        if not style_preferences and ai_attributes and "style" in ai_attributes:
            style_preferences = ai_attributes["style"]

        # 날씨 API 호출 (재시도 및 폴백)
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        fallback_weather = {
            "temperature": 23.5,
            "condition": "맑음",
            "humidity": 60,
            "wind_speed": 5.2,
        }
        used_fallback_weather = False
        try:
            session = requests.Session()
            retry = Retry(total=3, backoff_factor=0.3)
            session.mount("https://", HTTPAdapter(max_retries=retry))
            weather_api_obj = current_app.config.get("weather_api")
            weather_info = weather_api_obj.get_weather_info(latitude, longitude)
        except Exception as weather_err:
            current_app.logger.warning(f"날씨 API 실패: {weather_err}")
            weather_info = fallback_weather
            used_fallback_weather = True

        # 추천 생성
        try:
            db_images = current_app.config.get("DB_IMAGES", [])
            recommendation_result = final_recommendation(
                weather=weather_info,
                user_prompt=user_request,
                style_preferences=style_preferences,
                ai_attributes=ai_attributes,
                gemini_api_key=None,
                db_images=db_images,
            )
            response = {
                "success": True,
                "weather": weather_info,
                "recommendation_text": recommendation_result.get(
                    "recommendation_text", "추천을 생성했습니다."
                ),
                "suggested_items": recommendation_result.get(
                    "categories", ["반팔티", "청바지"]
                ),
                "ai_analysis": ai_attributes,
                "ai_debug_details": ai_debug_details,
                "recommendation_details": recommendation_result,
            }
            if used_fallback_weather:
                response["weather_fallback"] = True
            if ai_attributes is None:
                response["ai_analysis_status"] = (
                    "AI 분석 결과 없음 (상세: ai_debug_details 참조)"
                )

            return jsonify(response), 200

        except Exception as e:
            current_app.logger.error(f"추천 생성 실패: {e}")
            # 기본 폴백 응답
            return (
                jsonify(
                    {
                        "success": True,
                        "weather": weather_info,
                        "weather_fallback": used_fallback_weather,
                        "recommendation_text": f"오늘 날씨에는 {', '.join(style_preferences)} 스타일로 {user_request}에 맞는 코디를 추천합니다.",
                        "suggested_items": ["반팔티", "청바지", "스니커즈"],
                        "ai_analysis": ai_attributes,
                        "ai_debug_details": ai_debug_details,
                        "ai_analysis_status": "AI 분석 결과 없음 (상세: ai_debug_details 참조)",
                        "recommendation_details": {},
                    }
                ),
                200,
            )

    @app.route("/api/debug/ai-status", methods=["GET"])
    @swag_from(debug_ai_status_schema)
    def debug_ai_status():
        status = {
            "models_initialized": _models_initialized,
            "yolo_model_loaded": bool(yolo_model),
            "resnet_model_loaded": bool(resnet_model),
            "YOLOv11MultiTask_available": YOLOv11MultiTask is not None,
        }
        try:
            model_path = MODEL_PATHS.get("yolo")
            status["model_path"] = model_path
            status["model_file_exists"] = (
                os.path.exists(model_path) if model_path else False
            )
        except Exception as e:
            status["config_error"] = str(e)
        return jsonify(status), 200

    @app.route("/api/debug/weather-test", methods=["GET"])
    @swag_from(debug_weather_test_schema)
    def debug_weather_test():
        lat = request.args.get("lat", 37.5665, type=float)
        lon = request.args.get("lon", 126.9780, type=float)
        try:
            weather_data = weather_api.get_weather_info(lat, lon)
            return (
                jsonify(
                    {
                        "success": True,
                        "location": {"latitude": lat, "longitude": lon},
                        "weather_data": weather_data,
                        "timestamp": datetime.now().isoformat(),
                        "api_key_status": (
                            "OK"
                            if getattr(weather_api, "service_key_decoded", False)
                            else "Missing"
                        ),
                    }
                ),
                200,
            )
        except Exception as e:
            return (
                jsonify(
                    {
                        "success": False,
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                    }
                ),
                500,
            )

    @app.route("/api/dataset/by-style", methods=["GET"])
    def get_dataset_by_style():
        style = request.args.get("style")
        if not style:
            return jsonify({"error": "style 파라미터가 필요합니다."}), 400

        labels_dir = current_app.config["LABELS_DIR"]
        image_dir = current_app.config["IMAGE_DIR"]
        results = []

        for fname in os.listdir(labels_dir):
            if not fname.endswith(".json"):
                continue
            try:
                with open(os.path.join(labels_dir, fname), encoding="utf-8") as f:
                    data = json.load(f)
                style_list = (
                    data.get("데이터셋 정보", {}).get("라벨링", {}).get("스타일", [])
                )
                if any(s.get("스타일") == style for s in style_list):
                    image_name = data.get("이미지 정보", {}).get("이미지 파일명")
                    image_path = os.path.join(image_dir, image_name)
                    if os.path.exists(image_path):
                        results.append({"image": image_name, "label": fname})
            except Exception:
                continue

        return jsonify(results), 200

    return app


def initialize_ai_models(app=None):
    """
    YOLO와 ResNet 모델을 초기화합니다.
    """
    global yolo_model, resnet_model, _models_initialized
    if YOLOv11MultiTask is None or FashionAttributePredictor is None:
        msg = "YOLO/ResNet 모듈 import 실패"
        if app:
            app.logger.error(msg)
        else:
            print(msg)
        return False

    try:
        from ultralytics import YOLO
        import torch

        yolo_path = MODEL_PATHS.get("yolo")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not os.path.exists(yolo_path):
            msg = f"YOLO 모델 파일을 찾을 수 없습니다: {yolo_path}"
            if app:
                app.logger.error(msg)
            else:
                print(msg)
            return False

        yolo_raw = YOLO(yolo_path)
        yolo_model = YOLOv11MultiTask(yolo_raw)
        resnet_model = FashionAttributePredictor(device=device)

        _models_initialized = True
        if app:
            app.logger.info(
                f"YOLO/ResNet 모델이 성공적으로 로드되었습니다. (Device: {device})"
            )
        else:
            print(f"YOLO/ResNet 모델이 성공적으로 로드되었습니다. (Device: {device})")
        return True

    except Exception as e:
        msg = f"AI 모델 초기화 실패: {e}"
        if app:
            app.logger.error(msg)
        else:
            print(msg)
        return False


# --- Helper: 이미지 파일 파싱 ---
def parse_images(field_names=("images", "image")):
    for name in field_names:
        files = request.files.getlist(name)
        if files:
            return files
    return []


def combine_multiple_image_results(results_list):
    """
    여러 이미지 분석 결과를 통합하여 가장 많이 등장한 속성 위주로 반환
    """
    from collections import Counter

    valid_results = [r for r in results_list if r.get("attributes") is not None]
    if not valid_results:
        return None
    combined = {}
    for r in valid_results:
        for k, v in r["attributes"].items():
            combined.setdefault(k, [])
            if isinstance(v, list):
                for item in v:
                    combined[k].append(
                        item.get("class_name") if isinstance(item, dict) else str(item)
                    )
            else:
                combined[k].append(str(v))
    # 최빈값 위주로 정리
    summary = {}
    for k, vlist in combined.items():
        if not vlist:
            continue
        counter = Counter(vlist)
        summary[k] = [item for item, _ in counter.most_common(3)]
    return summary
