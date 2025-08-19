# backend/app/services.py
"""
AI 서비스 모듈 (간략화 버전)
- AI 모델 초기화 및 관리
- 이미지 분석 서비스
- 날씨 정보 서비스
"""
# Standard library imports
import logging
import os
from datetime import datetime
from typing import Optional, Tuple, Any, Dict, List

# Third-party imports
from flask import current_app

# Local imports
from config import Config
from app.error_codes import AIErrorCodes, create_error_response
from app.monitoring import performance_timer, monitor_performance

logger = logging.getLogger(__name__)

# 의존 모듈 로드 (선택적 import)
try:
    from app.ai.yolo_classification import YOLOv11MultiTask
    from app.ai.resnet_multi_classification import FashionAttributePredictor
    from app.recommender.final_recommender import final_recommendation
    from app.weather import KoreaWeatherAPI
    from app.utils import preprocess_image

    AI_MODULES_LOADED = True
    logger.info("모든 AI 모듈이 성공적으로 로드되었습니다.")
except ImportError as e:
    logger.error(f"AI 모듈 import 실패: {e}")
    YOLOv11MultiTask = None
    FashionAttributePredictor = None
    KoreaWeatherAPI = None
    final_recommendation = None
    AI_MODULES_LOADED = False


class AIModelManager:
    """AI 모델 관리 클래스 (간략화 버전)"""

    def __init__(self):
        self.yolo_model = None
        self.resnet_model = None
        self.weather_api_instance = None
        self._models_initialized = False
        self._model_load_time = None
        self._device = None

    @property
    def is_initialized(self) -> bool:
        """모델 초기화 상태 반환"""
        return self._models_initialized

    @property
    def device(self) -> str:
        """사용 중인 디바이스 반환"""
        return self._device or "unknown"

    def initialize_models(self, app):
        """AI 모델들을 초기화합니다."""
        if not AI_MODULES_LOADED:
            app.logger.error("AI 모듈이 로드되지 않아 초기화할 수 없습니다.")
            return False

        try:
            # 날씨 API 초기화
            self._initialize_weather_api(app)

            # AI 모델 초기화
            self._initialize_ai_models(app)

            return self._models_initialized

        except Exception as e:
            app.logger.error(f"모델 초기화 중 예외 발생: {e}", exc_info=True)
            self._models_initialized = False
            return False

    def _initialize_weather_api(self, app):
        """날씨 API를 초기화합니다."""
        if KoreaWeatherAPI:
            self.weather_api_instance = KoreaWeatherAPI()
            app.logger.info("날씨 API 인스턴스 생성 완료")
        else:
            app.logger.warning("날씨 API 모듈을 사용할 수 없습니다.")

    def _initialize_ai_models(self, app):
        """AI 모델들을 초기화합니다."""
        if not (YOLOv11MultiTask and FashionAttributePredictor):
            app.logger.error("YOLO 또는 ResNet 모듈이 로드되지 않았습니다.")
            return

        try:
            from ultralytics import YOLO
            import torch

            # 디바이스 설정
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            app.logger.info(f"AI 모델 디바이스: {self._device}")

            # GPU 메모리 초기화 (CUDA 사용시)
            if self._device == "cuda":
                torch.cuda.empty_cache()
                app.logger.info("GPU 메모리 초기화 완료")

            # YOLO 모델 경로 검증
            yolo_path = Config.MODEL_PATHS.get("yolo")
            if not yolo_path or not os.path.exists(yolo_path):
                raise FileNotFoundError(
                    f"YOLO 모델 파일을 찾을 수 없습니다: {yolo_path}"
                )

            # 모델 로딩
            with performance_timer("yolo_model_loading"):
                yolo_raw = YOLO(yolo_path)
                self.yolo_model = YOLOv11MultiTask(yolo_raw)

            with performance_timer("resnet_model_loading"):
                self.resnet_model = FashionAttributePredictor(device=self._device)

            self._models_initialized = True
            self._model_load_time = datetime.now()

            app.logger.info(f"AI 모델 초기화 완료 (디바이스: {self._device})")

        except Exception as e:
            app.logger.error(f"AI 모델 초기화 실패: {e}", exc_info=True)
            self._models_initialized = False

    def get_model_status(self) -> Dict[str, Any]:
        """모델 상태 정보를 반환합니다."""
        yolo_path = Config.MODEL_PATHS.get("yolo")
        resnet_path = Config.MODEL_PATHS.get("resnet")

        def get_file_info(file_path: Optional[str]) -> Dict[str, Any]:
            """파일 정보를 안전하게 수집"""
            if not file_path:
                return {"exists": False, "path": None, "size_mb": 0}

            try:
                if os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    return {
                        "exists": True,
                        "path": file_path,
                        "size_mb": round(file_size / (1024 * 1024), 2),
                    }
                else:
                    return {"exists": False, "path": file_path, "size_mb": 0}
            except Exception:
                return {"exists": False, "path": file_path, "size_mb": 0}

        yolo_file_info = get_file_info(yolo_path)
        resnet_file_info = get_file_info(resnet_path)

        return {
            "models_initialized": self._models_initialized,
            "yolo_model_loaded": bool(self.yolo_model),
            "resnet_model_loaded": bool(self.resnet_model),
            "device": self.device,
            "modules_available": {
                "YOLOv11MultiTask": YOLOv11MultiTask is not None,
                "FashionAttributePredictor": FashionAttributePredictor is not None,
                "KoreaWeatherAPI": KoreaWeatherAPI is not None,
                "final_recommendation": final_recommendation is not None,
            },
            "model_files": {
                "yolo": yolo_file_info,
                "resnet": resnet_file_info,
            },
            "model_load_time": (
                self._model_load_time.isoformat() if self._model_load_time else None
            ),
        }

    def clear_gpu_cache(self):
        """GPU 메모리 캐시를 정리합니다."""
        if self._device == "cuda":
            try:
                import torch

                torch.cuda.empty_cache()
                logger.debug("GPU 메모리 캐시 정리 완료")
                return True
            except Exception as e:
                logger.warning(f"GPU 캐시 정리 실패: {e}")
                return False
        return True

    def get_gpu_info(self) -> Dict[str, Any]:
        """GPU 상태 정보를 반환합니다."""
        gpu_info = {"device": self._device, "cuda_available": False}

        if self._device == "cuda":
            try:
                import torch

                gpu_info.update(
                    {
                        "cuda_available": True,
                        "cuda_device_count": torch.cuda.device_count(),
                        "current_device": torch.cuda.current_device(),
                        "device_name": (
                            torch.cuda.get_device_name(0)
                            if torch.cuda.device_count() > 0
                            else "Unknown"
                        ),
                        "memory_allocated_mb": round(
                            torch.cuda.memory_allocated() / 1024 / 1024, 2
                        ),
                        "memory_reserved_mb": round(
                            torch.cuda.memory_reserved() / 1024 / 1024, 2
                        ),
                    }
                )
            except Exception as e:
                logger.warning(f"GPU 정보 수집 실패: {e}")
                gpu_info["error"] = str(e)

        return gpu_info


# 전역 모델 매니저 인스턴스
model_manager = AIModelManager()


def initialize_ai_models(app):
    """AI 모델 초기화"""
    return model_manager.initialize_models(app)


def get_ai_model_status():
    """AI 모델 상태 반환"""
    return model_manager.get_model_status()


def get_gpu_status():
    """GPU 상태 정보 반환"""
    return model_manager.get_gpu_info()


def clear_gpu_memory():
    """GPU 메모리 정리"""
    return model_manager.clear_gpu_cache()


@monitor_performance("image_analysis")
def analyze_single_image(
    uploaded_file, file_idx: int
) -> Tuple[Optional[List[Dict]], Dict[str, Any]]:
    """업로드된 단일 이미지를 분석하고 속성을 추출합니다."""
    filename = uploaded_file.filename
    debug_info = {"id": file_idx, "filename": filename}

    if not model_manager.is_initialized:
        debug_info["error"] = "AI 모델이 초기화되지 않았습니다."
        debug_info["error_code"] = AIErrorCodes.MODEL_NOT_INITIALIZED
        return None, debug_info

    # 기본 파일 검증
    if not filename:
        debug_info["error"] = "파일명이 없습니다."
        debug_info["error_code"] = AIErrorCodes.INVALID_IMAGE_FORMAT
        return None, debug_info

    # MIME 타입 검증
    allowed_mimetypes = Config.INPUT_VALIDATION["allowed_mimetypes"]
    if uploaded_file.mimetype not in allowed_mimetypes:
        debug_info["error"] = f"지원되지 않는 이미지 형식: {uploaded_file.mimetype}"
        debug_info["error_code"] = AIErrorCodes.INVALID_IMAGE_FORMAT
        return None, debug_info

    try:
        # 이미지 전처리
        with performance_timer("image_preprocessing"):
            image_np = preprocess_image(uploaded_file)

        logger.debug(f"이미지 전처리 완료: {filename}")  # info -> debug

        # YOLO 객체 탐지
        yolo_results, crops = _perform_yolo_detection(image_np, filename)
        if not crops:
            debug_info["error"] = "탐지된 객체가 없습니다."
            debug_info["error_code"] = AIErrorCodes.IMAGE_PROCESSING_FAILED
            return _create_empty_result(file_idx, filename), debug_info

        # ResNet 속성 분석
        analysis_results = _perform_resnet_analysis(crops, file_idx, filename)

        debug_info["success"] = True
        debug_info["num_objects"] = len(crops)

        # 로그 레벨 최적화 - 객체 수가 많을 때만 info 로그
        if len(crops) >= 5:
            logger.info(f"다중 객체 분석 완료: {filename} ({len(crops)}개 객체)")
        else:
            logger.debug(f"이미지 분석 완료: {filename} ({len(crops)}개 객체)")

        return analysis_results, debug_info

    except Exception as e:
        logger.error(f"이미지 분석 중 오류 ({filename}): {e}", exc_info=True)
        debug_info["error"] = f"이미지 처리 오류: {e}"
        debug_info["error_code"] = AIErrorCodes.IMAGE_PROCESSING_FAILED
        return None, debug_info


def _perform_yolo_detection(image_np, filename: str) -> Tuple[Any, List]:
    """YOLO 객체 탐지를 수행합니다."""
    try:
        with performance_timer("yolo_inference"):
            yolo_config = Config.AI_MODEL_CONFIG.get("yolo", {})
            conf_threshold = yolo_config.get("conf_threshold", 0.4)
            iou_threshold = yolo_config.get("iou_threshold", 0.5)

            yolo_results = model_manager.yolo_model.detect(
                image_np, conf_thres=conf_threshold, iou_thres=iou_threshold
            )
            crops = model_manager.yolo_model.extract_crops(
                image_np, yolo_results, conf_thres=conf_threshold
            )

        logger.debug(f"YOLO 탐지 완료: {filename} ({len(crops)}개 객체)")
        return yolo_results, crops

    except Exception as e:
        logger.error(f"YOLO 탐지 실패 ({filename}): {e}")
        raise


def _perform_resnet_analysis(
    crops: List, file_idx: int, filename: str
) -> List[Dict[str, Any]]:
    """ResNet을 사용하여 속성 분석을 수행합니다."""
    analysis_results = []

    for obj_idx, (crop, bbox, conf, cls) in enumerate(crops):
        category_name = _get_category_name(cls)

        try:
            with performance_timer("resnet_inference"):
                attributes = model_manager.resnet_model.predict_attributes(crop)

            bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)

            analysis_results.append(
                {
                    "id": file_idx,
                    "filename": filename,
                    "object_index": obj_idx,
                    "attributes": attributes,
                    "confidence": float(conf),
                    "detected_class": int(cls),
                    "bbox": bbox_list,
                    "category": category_name,
                }
            )

            logger.debug(
                f"ResNet 분석 완료: {filename} obj_{obj_idx} -> {category_name}"
            )

        except Exception as e:
            logger.error(f"ResNet 분석 실패 ({filename}, obj_{obj_idx}): {e}")

            try:
                bbox_list = bbox.tolist() if hasattr(bbox, "tolist") else list(bbox)
            except Exception:
                bbox_list = [0, 0, 0, 0]

            analysis_results.append(
                {
                    "id": file_idx,
                    "filename": filename,
                    "object_index": obj_idx,
                    "error": f"ResNet 분석 실패: {str(e)[:100]}",
                    "error_code": AIErrorCodes.RESNET_MODEL_ERROR,
                    "confidence": float(conf) if conf is not None else 0.0,
                    "detected_class": int(cls) if cls is not None else -1,
                    "bbox": bbox_list,
                    "category": category_name,
                    "attributes": None,
                }
            )

    return analysis_results


def _get_category_name(cls) -> str:
    """클래스 인덱스를 카테고리 이름으로 변환합니다."""
    try:
        category_list = Config.CLASS_MAPPINGS.get("category", [])
        if isinstance(category_list, list) and int(cls) < len(category_list):
            return category_list[int(cls)]
        else:
            return f"class_{int(cls)}"
    except (ValueError, TypeError):
        return f"unknown_class_{cls}"


def _create_empty_result(file_idx: int, filename: str) -> List[Dict[str, Any]]:
    """빈 결과를 생성합니다."""
    return [
        {
            "id": file_idx,
            "filename": filename,
            "attributes": None,
            "confidence": 0.0,
            "detected_class": -1,
            "bbox": "No objects detected",
            "error_code": AIErrorCodes.IMAGE_PROCESSING_FAILED,
        }
    ]


@monitor_performance("weather_api_call")
def get_weather_info(latitude: float, longitude: float) -> Tuple[Dict[str, Any], bool]:
    """날씨 정보를 가져옵니다."""
    if not model_manager.weather_api_instance:
        logger.warning("날씨 API가 초기화되지 않아 기본 데이터를 사용합니다.")
        return {
            "temperature": 20.0,
            "condition": "알 수 없음",
            "humidity": 50,
            "wind_speed": 0.0,
            "error": "날씨 정보를 가져올 수 없습니다.",
        }, True

    try:
        weather_info = model_manager.weather_api_instance.get_weather_info(
            latitude, longitude
        )
        logger.debug(f"날씨 정보 조회 성공: 위도={latitude}, 경도={longitude}")
        return weather_info, False

    except Exception as weather_err:
        logger.warning(f"날씨 API 호출 실패: {weather_err}. 기본 데이터를 사용합니다.")
        return {
            "temperature": 20.0,
            "condition": "알 수 없음",
            "humidity": 50,
            "wind_speed": 0.0,
            "error_code": AIErrorCodes.WEATHER_API_FAILED,
            "error": "날씨 정보를 가져올 수 없습니다.",
        }, True


def get_final_recommendation(*args, **kwargs):
    """final_recommendation 함수에 대한 wrapper"""
    if final_recommendation is None:
        raise ImportError("final_recommendation 모듈이 로드되지 않았습니다.")
    return final_recommendation(*args, **kwargs)
