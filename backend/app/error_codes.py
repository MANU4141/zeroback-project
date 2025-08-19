"""
AI 기능 표준 에러 코드 정의 및 응답 유틸리티 (프로토타입 단순화 버전)
"""

from datetime import datetime
from typing import Any, Dict, Tuple


def now_iso() -> str:
    """간단 ISO 포맷 UTC 시간"""
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


# 최소 에러 코드 정의 (필요한 것만)
class AIErrorCodes:
    MODEL_NOT_INITIALIZED = "AI-002"
    IMAGE_PROCESSING_FAILED = "AI-010"
    RESNET_MODEL_ERROR = "AI-013"
    INVALID_IMAGE_FORMAT = "AI-101"
    WEATHER_API_FAILED = "AI-020"
    INTERNAL_SERVER_ERROR = "AI-301"


# 에러 코드 기본 HTTP 상태 매핑 (간략판)
DEFAULT_HTTP_STATUS = {
    AIErrorCodes.MODEL_NOT_INITIALIZED: 500,
    AIErrorCodes.IMAGE_PROCESSING_FAILED: 500,
    AIErrorCodes.RESNET_MODEL_ERROR: 500,
    AIErrorCodes.INVALID_IMAGE_FORMAT: 400,
    AIErrorCodes.WEATHER_API_FAILED: 502,
    AIErrorCodes.INTERNAL_SERVER_ERROR: 500,
}


def create_error_response(
    error_code: str,
    message: str,
    http_status: int = None,
    details: Dict[str, Any] = None,
) -> Tuple[Dict[str, Any], int]:
    """단순 에러 응답"""
    status = http_status or DEFAULT_HTTP_STATUS.get(error_code, 500)
    body = {
        "success": False,
        "error_code": error_code,
        "message": message,
        "timestamp": now_iso(),
    }
    if details:
        body["details"] = details
    return body, status


def create_success_response(data: Any, message: str = "Success") -> Dict[str, Any]:
    """단순 성공 응답"""
    return {
        "success": True,
        "message": message,
        "data": data,
        "timestamp": now_iso(),
    }
