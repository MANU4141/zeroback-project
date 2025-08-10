import os
import logging
import time
from flask import current_app

# 의존 모듈 로드
try:
    from backend.app.ai.yolo_multitask import YOLOv11MultiTask
    from backend.app.ai.resnet_multitask import FashionAttributePredictor
    from backend.app.recommender.final_recommender import final_recommendation
    from backend.app.weather import KoreaWeatherAPI
    from backend.config import Config
    from backend.app.utils import preprocess_image
except ImportError as e:
    logging.basicConfig(level=logging.INFO)
    logging.error(f"서비스 모듈 import 실패: {e}")
    YOLOv11MultiTask = None
    FashionAttributePredictor = None
    KoreaWeatherAPI = None
    final_recommendation = None

# 전역 변수
yolo_model = None
resnet_model = None
weather_api_instance = None
_models_initialized = False


def initialize_ai_models(app):
    """AI 모델(YOLO, ResNet) 및 날씨 API를 초기화합니다."""
    global yolo_model, resnet_model, weather_api_instance, _models_initialized

    # 날씨 API 초기화
    if KoreaWeatherAPI:
        weather_api_instance = KoreaWeatherAPI()
        app.logger.info("날씨 API 인스턴스가 생성되었습니다.")
    else:
        app.logger.warning("날씨 API 모듈이 로드되지 않았습니다.")

    # AI 모델 초기화
    if YOLOv11MultiTask is None or FashionAttributePredictor is None:
        app.logger.error("YOLO/ResNet 모듈이 import되지 않았습니다.")
        _models_initialized = False
        return

    try:
        from ultralytics import YOLO
        import torch

        yolo_path = Config.MODEL_PATHS.get("yolo")
        device = "cuda" if torch.cuda.is_available() else "cpu"

        if not yolo_path or not os.path.exists(yolo_path):
            app.logger.error(f"YOLO 모델 파일을 찾을 수 없습니다: {yolo_path}")
            _models_initialized = False
            return

        yolo_raw = YOLO(yolo_path)
        yolo_model = YOLOv11MultiTask(yolo_raw)
        resnet_model = FashionAttributePredictor(device=device)
        _models_initialized = True
        app.logger.info(
            f"YOLO/ResNet 모델이 성공적으로 로드되었습니다. (Device: {device})"
        )
    except Exception as e:
        app.logger.error(f"AI 모델 초기화 실패: {e}", exc_info=True)
        _models_initialized = False


def get_ai_model_status():
    """AI 모델의 현재 상태를 반환합니다."""
    from backend.config import Config

    yolo_path = getattr(Config, "MODEL_PATHS", {}).get("yolo")
    status = {
        "models_initialized": _models_initialized,
        "yolo_model_loaded": bool(yolo_model),
        "resnet_model_loaded": bool(resnet_model),
        # 확장 상태 정보 (구/신 클라이언트 호환)
        "YOLOv11MultiTask_available": YOLOv11MultiTask is not None,
        "model_file_exists": bool(yolo_path) and os.path.exists(yolo_path),
        "yolo_model_instance": type(yolo_model).__name__ if yolo_model else None,
        "resnet_model_instance": type(resnet_model).__name__ if resnet_model else None,
    }
    return status


def analyze_single_image(uploaded_file, file_idx):
    """업로드된 단일 이미지를 분석하고 속성을 추출합니다."""
    filename = uploaded_file.filename
    debug_info = {"id": file_idx, "filename": filename}

    # 모델 예측 시간 측정 시작
    model_start_time = time.time()

    # 기본 파일 검증
    if not filename:
        debug_info["error"] = "No filename"
        return None, debug_info
    if uploaded_file.mimetype not in ["image/jpeg", "image/png"]:
        debug_info["error"] = "지원되지 않는 이미지 형식"
        return None, debug_info

    try:
        # 이미지 전처리 시간
        preprocess_start = time.time()
        image_np = preprocess_image(uploaded_file)
        preprocess_time = time.time() - preprocess_start
        current_app.logger.info(
            f"[AI] 이미지 전처리 완료: {filename} ({preprocess_time:.3f}초)"
        )

        # 1) YOLO 탐지 시간 측정
        yolo_start = time.time()
        try:
            yolo_results = yolo_model.detect(image_np)
            crops = yolo_model.extract_crops(image_np, yolo_results)
            yolo_time = time.time() - yolo_start

            if not crops:
                raise ValueError("객체 없음")
            current_app.logger.info(
                f"[YOLO] {len(crops)}개 객체 탐지: {filename} ({yolo_time:.3f}초)"
            )
        except Exception as yolo_err:
            yolo_time = time.time() - yolo_start
            debug_info["error"] = f"YOLO 탐지 실패: {yolo_err}"
            debug_info["processing_time"] = time.time() - model_start_time
            debug_info["yolo_time"] = yolo_time
            current_app.logger.error(f"[YOLO] 탐지 실패 ({filename}): {yolo_err}")
            return [
                {
                    "id": file_idx,
                    "filename": filename,
                    "attributes": None,
                    "confidence": 0.0,
                    "detected_class": -1,
                    "bbox": "No objects detected",
                }
            ], debug_info

        analysis_results = []
        total_resnet_time = 0

        for obj_idx, (crop, bbox, conf, cls) in enumerate(crops):
            # 카테고리 이름 매핑 (범위를 벗어나면 클래스 인덱스 문자열 사용)
            category_list = Config.CLASS_MAPPINGS.get("category", [])
            try:
                category_name = (
                    category_list[int(cls)]
                    if isinstance(category_list, list) and int(cls) < len(category_list)
                    else str(int(cls))
                )
            except Exception:
                category_name = str(int(cls))

            current_app.logger.info(
                f"[YOLO] file={filename}, obj={obj_idx}, class={category_name}, conf={conf:.2f}"
            )

            # 2) ResNet 속성 예측 시간 측정
            resnet_start = time.time()
            try:
                attributes = resnet_model.predict_attributes(crop)
                resnet_time = time.time() - resnet_start
                total_resnet_time += resnet_time
                current_app.logger.info(
                    f"[ResNet] 속성 예측 완료 ({filename}, obj={obj_idx}) ({resnet_time:.3f}초)"
                )

                # bbox를 리스트로 안전 변환
                bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
                analysis_results.append(
                    {
                        "id": file_idx,
                        "filename": filename,
                        "attributes": attributes,
                        "confidence": float(conf),
                        "detected_class": int(cls),
                        "bbox": bbox_list,
                        "category": category_name,
                    }
                )
            except Exception as resnet_err:
                current_app.logger.error(
                    f"[ResNet] 예측 실패 (file: {filename}, obj: {obj_idx}): {resnet_err}"
                )
                bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
                analysis_results.append(
                    {
                        "id": file_idx,
                        "filename": filename,
                        "error": f"ResNet failed: {resnet_err}",
                        "confidence": float(conf),
                        "detected_class": int(cls),
                        "bbox": bbox_list,
                        "category": category_name,
                        "attributes": None,
                    }
                )

        # 전체 모델 예측 시간 계산
        total_model_time = time.time() - model_start_time

        # 시간 정보를 debug_info에 추가
        debug_info["success"] = True
        debug_info["processing_time"] = total_model_time
        debug_info["preprocess_time"] = preprocess_time
        debug_info["yolo_time"] = yolo_time
        debug_info["resnet_total_time"] = total_resnet_time
        debug_info["num_objects"] = len(crops)

        current_app.logger.info(
            f"[타이밍] {filename} 총 처리시간: {total_model_time:.3f}초 "
            f"(전처리: {preprocess_time:.3f}초, YOLO: {yolo_time:.3f}초, "
            f"ResNet: {total_resnet_time:.3f}초, 객체수: {len(crops)}개)"
        )

        return analysis_results, debug_info

    except Exception as e:
        total_model_time = time.time() - model_start_time
        current_app.logger.error(
            f"이미지 분석 중 오류 (file: {filename}): {e}", exc_info=True
        )
        return None, {
            "id": file_idx,
            "filename": filename,
            "error": f"이미지 처리 오류: {e}",
            "processing_time": total_model_time,
        }


def get_weather_info(latitude, longitude):
    """날씨 정보를 가져옵니다 (폴백 기능 포함)."""
    if not weather_api_instance:
        current_app.logger.warning(
            "날씨 API가 초기화되지 않아 폴백 데이터를 사용합니다."
        )
        # 인스턴스가 없을 때도 안전하게 폴백 값 반환
        try:
            return KoreaWeatherAPI().get_fallback_weather(), True
        except Exception:
            return {
                "temperature": 23.5,
                "condition": "맑음",
                "humidity": 60,
                "wind_speed": 5.0,
            }, True

    try:
        weather_info = weather_api_instance.get_weather_info(latitude, longitude)
        return weather_info, False
    except Exception as weather_err:
        current_app.logger.warning(
            f"날씨 API 호출 실패: {weather_err}. 폴백 데이터를 사용합니다."
        )
        return weather_api_instance.get_fallback_weather(), True
