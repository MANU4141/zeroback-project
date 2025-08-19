# backend/config.py
import os
import json
import logging
from dotenv import load_dotenv

# .env 파일이 있다면 로드합니다.
load_dotenv()

# 로깅 설정
logger = logging.getLogger(__name__)


def _parse_bool_env(env_var: str, default: bool = False) -> bool:
    """환경변수를 boolean으로 안전하게 파싱"""
    value = os.getenv(env_var, str(default)).strip().lower()
    return value in ("true", "1", "yes", "on")


def _parse_int_env(env_var: str, default: int) -> int:
    """환경변수를 int로 안전하게 파싱"""
    try:
        return int(os.getenv(env_var, str(default)))
    except ValueError:
        logger.warning(f"환경변수 {env_var} 파싱 실패, 기본값 {default} 사용")
        return default


def _parse_list_env(env_var: str, default: list, separator: str = ",") -> list:
    """환경변수를 리스트로 안전하게 파싱"""
    value = os.getenv(env_var)
    if not value:
        return default
    return [item.strip() for item in value.split(separator) if item.strip()]


class Config:
    """애플리케이션 환경 설정을 담는 클래스"""

    # --- 기본 경로 설정 ---
    # 이 파일의 위치(__file__)를 기준으로 BASE_DIR를 설정합니다.
    # backend 폴더의 절대 경로가 됩니다.
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # --- 모델 경로 ---
    # AI 추론에 사용되는 학습된 모델 파일들의 경로
    MODEL_PATHS = {
        "yolo": os.path.join(BASE_DIR, "models", "YOLOv11_large.pt"),  # 객체 검출용
        "resnet": os.path.join(BASE_DIR, "models", "ResNet50.pth"),  # 속성 분류용
    }

    # --- 데이터셋 경로 ---
    # 추천에 사용되는 이미지 데이터베이스와 라벨 정보
    IMAGE_DIR = os.path.join(BASE_DIR, "DATA", "images")  # 패션 이미지 저장소
    LABELS_DIR = os.path.join(BASE_DIR, "DATA", "labels_json")  # 라벨/메타데이터

    # --- API 키 ---
    # os.getenv를 사용하여 환경 변수에서 API 키를 가져옵니다.
    WEATHER_API_KEY_ENCODE = os.getenv("WEATHER_API_KEY_ENCODE")
    WEATHER_API_KEY_DECODE = os.getenv("WEATHER_API_KEY_DECODE")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

    # --- 클래스 매핑 로드 ---
    @classmethod
    def _load_class_mappings(cls):
        """클래스 매핑을 JSON 파일에서 로드합니다."""
        try:
            mappings_path = os.path.join(cls.BASE_DIR, "config", "class_mappings.json")
            with open(mappings_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            # 폴백: 기본 클래스 매핑 제공
            logger.warning(
                f"클래스 매핑 파일 로드 실패({mappings_path}): {e}. 기본값을 사용합니다."
            )
            return {
                "category": ["탑", "청바지", "드레스"],
                "color": ["블랙", "화이트", "블루"],
                "style": ["캐주얼", "클래식", "모던"],
            }

    @classmethod
    def validate_config(cls, strict_mode: bool = False):
        """설정값의 유효성을 검증합니다.

        Args:
            strict_mode: True일 경우 중요한 오류 시 시스템 종료

        Returns:
            list: 발견된 오류 목록
        """
        errors = []
        critical_errors = []

        # 모델 파일 존재 확인
        for model_name, model_path in cls.MODEL_PATHS.items():
            if not os.path.exists(model_path):
                error_msg = f"{model_name} 모델 파일이 존재하지 않습니다: {model_path}"
                errors.append(error_msg)
                if model_name in ["yolo", "resnet"]:  # 핵심 모델
                    critical_errors.append(error_msg)

        # 디렉토리 존재 확인
        for dir_name, dir_path in [
            ("IMAGE_DIR", cls.IMAGE_DIR),
            ("LABELS_DIR", cls.LABELS_DIR),
        ]:
            if not os.path.exists(dir_path):
                error_msg = f"{dir_name} 디렉토리가 존재하지 않습니다: {dir_path}"
                errors.append(error_msg)
                if dir_name == "IMAGE_DIR":  # 이미지 디렉토리는 중요
                    critical_errors.append(error_msg)

        # API 키 확인
        if not cls.WEATHER_API_KEY_DECODE:
            errors.append("WEATHER_API_KEY_DECODE가 설정되지 않았습니다.")
        if not cls.GEMINI_API_KEY:
            critical_errors.append("GEMINI_API_KEY가 설정되지 않았습니다.")

        # 결과 로깅 및 처리
        if critical_errors and strict_mode:
            logger.error(
                "중요한 설정 오류가 발견되어 시스템을 종료합니다:\n"
                + "\n".join(f"  - {error}" for error in critical_errors)
            )
            raise SystemExit(1)

        if errors:
            logger.warning(
                "설정 검증에서 다음 문제가 발견되었습니다:\n"
                + "\n".join(f"  - {error}" for error in errors)
            )
        else:
            logger.info("설정 검증 완료")

        return errors

    CLASS_MAPPINGS = None  # 나중에 초기화됩니다

    # --- AI 모델 추론 설정 ---
    AI_MODEL_CONFIG = {
        "yolo": {
            "conf_threshold": 0.4,  # 신뢰도 임계값 (낮을수록 더 많은 박스 검출)
            "iou_threshold": 0.5,  # IoU 임계값 (Non-Maximum Suppression)
            "max_detections": 4,  # 최대 검출 개수
        },
        "resnet": {
            "batch_size": 32,  # 배치 크기
            "num_workers": 4,  # 데이터 로더 워커 수
        },
    }

    # --- 입력 검증 설정 ---
    INPUT_VALIDATION = {
        "max_file_size_mb": 5,  # 최대 파일 크기 (MB)
        "min_resolution": 224,  # 최소 해상도 (px)
        "allowed_mimetypes": [  # 정렬된 리스트로 일관성 확보 (프론트/백엔드 동일 값 유지 권장)
            "image/bmp",
            "image/gif",
            "image/jpeg",
            "image/png",
            "image/webp",
        ],
        "max_images_per_request": 5,  # 요청당 최대 이미지 수
        "coordinate_bounds": {  # 좌표 유효 범위 (한국)
            "lat_min": 33.0,
            "lat_max": 38.9,
            "lng_min": 124.0,
            "lng_max": 132.0,
        },
    }

    # --- CORS 및 Flask 실행 설정 ---
    CORS_ORIGINS = os.getenv("CORS_ORIGINS", "*")
    FLASK_RUN_PORT = _parse_int_env("FLASK_RUN_PORT", 5000)
    FLASK_RUN_HOST = os.getenv("FLASK_RUN_HOST", "0.0.0.0")
    FLASK_DEBUG = _parse_bool_env("FLASK_DEBUG", False)


# 클래스 정의 후 CLASS_MAPPINGS 초기화
Config.CLASS_MAPPINGS = Config._load_class_mappings()
