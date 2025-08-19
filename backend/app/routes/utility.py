# backend/app/routes/utility.py
"""
유틸리티 관련 라우트
"""
import os
import logging
from flask import current_app, send_file, jsonify
from flasgger import swag_from

from app.schemas import serve_image_schema
from app.error_codes import create_error_response, AIErrorCodes
from config import Config

logger = logging.getLogger(__name__)


def register_utility_routes(app):
    """유틸리티 관련 라우트 등록"""

    @app.route("/api/images/<filename>", methods=["GET"])
    @swag_from(serve_image_schema)
    def serve_image(filename):
        """이미지 파일을 제공하는 엔드포인트"""
        try:
            # 파일명 검증
            if not filename or ".." in filename or "/" in filename or "\\" in filename:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error_code": "VALIDATION_ERROR",
                            "message": "유효하지 않은 파일명입니다.",
                            "details": {"field": "filename", "value": filename},
                        }
                    ),
                    400,
                )

            # 허용된 확장자 검증
            allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
            file_ext = os.path.splitext(filename.lower())[1]

            if file_ext not in allowed_extensions:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error_code": "VALIDATION_ERROR",
                            "message": f"지원되지 않는 파일 형식입니다. 지원 형식: {', '.join(allowed_extensions)}",
                            "details": {"field": "filename", "value": filename},
                        }
                    ),
                    400,
                )

            # 이미지 파일 경로 구성
            image_path = os.path.join(Config.IMAGE_DIR, filename)

            # 파일 존재 확인
            if not os.path.exists(image_path):
                return (
                    jsonify(
                        {
                            "success": False,
                            "error_code": AIErrorCodes.FILE_SYSTEM_ERROR,
                            "message": "요청한 이미지 파일을 찾을 수 없습니다.",
                            "details": {"filename": filename},
                        }
                    ),
                    404,
                )

            # 파일 제공
            return send_file(image_path, as_attachment=False, download_name=filename)

        except Exception as e:
            logger.exception(f"이미지 제공 중 오류 ({filename})")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": AIErrorCodes.FILE_SYSTEM_ERROR,
                        "message": "이미지 파일 제공 중 오류가 발생했습니다.",
                        "details": {"filename": filename, "error": str(e)},
                    }
                ),
                500,
            )

    @app.route("/", methods=["GET"])
    def index():
        """루트 경로 - API 정보 제공"""
        api_info = {
            "name": "Zeroback Fashion AI API",
            "version": "1.0.0",
            "description": "AI 기반 패션 추천 서비스",
            "endpoints": {
                "health": "/api/health",
                "recommend": "/api/recommend",
                "docs": "/docs/",
                "debug": {
                    "ai_status": "/api/debug/ai-status",
                    "weather_test": "/api/debug/weather-test",
                    "performance": "/api/debug/performance",
                },
            },
            "status": "running",
        }

        return jsonify(
            {
                "success": True,
                "message": "Zeroback Fashion AI API에 오신 것을 환영합니다!",
                "data": api_info,
            }
        )

    @app.route("/swagger", methods=["GET"])
    def swagger_redirect():
        """Swagger UI로 리다이렉트"""
        return """
        <!DOCTYPE html>
        <html>
        <head>
            <title>API Documentation</title>
            <meta http-equiv="refresh" content="0; url=/docs/">
        </head>
        <body>
            <p>API 문서로 이동 중... <a href="/docs/">여기를 클릭하세요</a></p>
        </body>
        </html>
        """
