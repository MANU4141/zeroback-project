# backend/app/routes/health.py
"""
헬스체크 관련 라우트
"""
# Standard library imports
from datetime import datetime

# Third-party imports
import torch
from flask import current_app, jsonify
from flasgger import swag_from

# Local imports
from app.schemas import health_check_schema
from app.services import get_ai_model_status
from app.error_codes import AIErrorCodes, create_error_response
from app.monitoring import performance_monitor


def register_health_routes(app):
    """헬스체크 관련 라우트 등록"""

    @app.route("/api/health", methods=["GET"])
    @swag_from(health_check_schema)
    def health_check():
        """서버의 상태를 확인하는 엔드포인트입니다."""
        try:
            status = get_ai_model_status()

            # GPU 정보 수집
            gpu_info = _get_gpu_info()

            # 성능 정보 수집
            performance_info = _get_performance_info()

            # 전체 상태 판단
            overall_status = _determine_overall_status(status, performance_info)

            health_data = {
                "status": overall_status,
                "model_details": status,
                "system": {
                    "gpu_available": gpu_info["gpu_available"],
                    "gpu_count": gpu_info["gpu_count"],
                    "gpu_memory": gpu_info["gpu_memory_info"],
                },
                "performance": performance_info,
                "db_images_count": len(current_app.config.get("DB_IMAGES", [])),
            }

            return jsonify(
                {"success": True, "message": "서버 상태 조회 성공", "data": health_data}
            )

        except Exception as e:
            current_app.logger.exception("헬스체크 중 오류")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "AI-301",
                        "message": "헬스체크 실행 중 오류가 발생했습니다.",
                        "details": {"error": str(e)},
                    }
                ),
                500,
            )


def _get_gpu_info():
    """GPU 정보를 수집합니다."""
    try:
        gpu_available = torch.cuda.is_available()
        gpu_count = torch.cuda.device_count() if gpu_available else 0
        gpu_memory_info = None

        if gpu_available and gpu_count > 0:
            try:
                gpu_memory_info = {
                    "allocated_mb": round(
                        torch.cuda.memory_allocated(0) / (1024 * 1024), 2
                    ),
                    "reserved_mb": round(
                        torch.cuda.memory_reserved(0) / (1024 * 1024), 2
                    ),
                }
            except Exception:
                gpu_memory_info = {"error": "GPU 메모리 정보 수집 실패"}

        return {
            "gpu_available": gpu_available,
            "gpu_count": gpu_count,
            "gpu_memory_info": gpu_memory_info,
        }
    except ImportError:
        return {"gpu_available": False, "gpu_count": 0, "gpu_memory_info": None}


def _get_performance_info():
    """성능 정보를 수집합니다."""
    return {
        "metrics": performance_monitor.get_summary(),
    }


def _determine_overall_status(model_status, performance_info):
    """전체 시스템 상태를 판단합니다."""
    if not model_status.get("models_initialized", False):
        return "ERROR"

    if current_app.config.get("INITIALIZATION_ERROR"):
        return "WARNING"

    return "OK"
