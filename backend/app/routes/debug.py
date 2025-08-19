# backend/app/routes/debug.py
"""
디버그 관련 라우트
"""
import logging
from flask import request, jsonify
from flasgger import swag_from

from app.schemas import debug_ai_status_schema, debug_weather_test_schema
from app.services import get_ai_model_status, get_weather_info, analyze_single_image
from app.error_codes import AIErrorCodes
from app.utils import validate_coordinates
from app.monitoring import performance_monitor

logger = logging.getLogger(__name__)


# 공통 응답 헬퍼 함수들
def _ok(data, message="Success"):
    """성공 응답 헬퍼"""
    return jsonify(
        {
            "success": True,
            "data": data,
            "message": message,
            "timestamp": __import__("datetime").datetime.now().isoformat(),
        }
    )


def _bad(error_code, message, details=None):
    """에러 응답 헬퍼"""
    from app.error_codes import create_error_response

    error_response, status_code = create_error_response(
        error_code=error_code, message=message, details=details
    )
    return jsonify(error_response), status_code


def _invalid(field, value, reason):
    """검증 에러 응답 헬퍼"""
    return (
        jsonify(
            {
                "success": False,
                "error_code": AIErrorCodes.VALIDATION_ERROR,
                "message": f"잘못된 {field}: {reason}",
                "details": {"field": field, "value": value, "reason": reason},
                "timestamp": __import__("datetime").datetime.now().isoformat(),
            }
        ),
        400,
    )


def register_debug_routes(app):
    """디버그 관련 라우트 등록"""

    @app.route("/api/debug/ai-status", methods=["GET"])
    @swag_from(debug_ai_status_schema)
    def debug_ai_status():
        """AI 모델 상태를 반환하는 디버그 엔드포인트"""
        try:
            status = get_ai_model_status()
            return _ok(status, "AI 모델 상태 조회 성공")
        except Exception as e:
            logger.exception(f"AI 상태 조회 중 오류: {e}")
            return _bad(
                AIErrorCodes.INTERNAL_SERVER_ERROR,
                "AI 상태 조회 중 오류가 발생했습니다.",
                {"error": str(e)},
            )

    @app.route("/api/debug/test-predict", methods=["POST"])
    def test_predict():
        """이미지 분석 테스트 엔드포인트"""
        try:
            uploaded_files = request.files.getlist("images") or request.files.getlist(
                "image"
            )

            if not uploaded_files or not uploaded_files[0].filename:
                return _invalid("images", None, "테스트할 이미지 파일이 필요합니다.")

            results = []
            for idx, uploaded_file in enumerate(
                uploaded_files[:3]
            ):  # 최대 3개만 테스트
                try:
                    analysis_result, debug_info = analyze_single_image(
                        uploaded_file, idx
                    )
                    results.append(
                        {
                            "file_index": idx,
                            "filename": uploaded_file.filename,
                            "analysis_result": analysis_result,
                            "debug_info": debug_info,
                        }
                    )
                except Exception as e:
                    results.append(
                        {
                            "file_index": idx,
                            "filename": uploaded_file.filename,
                            "error": str(e),
                            "error_code": AIErrorCodes.IMAGE_PROCESSING_FAILED,
                        }
                    )

            return _ok(
                {"test_results": results}, f"{len(results)}개 이미지 테스트 완료"
            )

        except Exception as e:
            logger.exception(f"이미지 분석 테스트 중 오류: {e}")
            return _bad(
                AIErrorCodes.INTERNAL_SERVER_ERROR,
                "이미지 분석 테스트 중 오류가 발생했습니다.",
                {"error": str(e)},
            )

    @app.route("/api/debug/weather-test", methods=["GET"])
    @swag_from(debug_weather_test_schema)
    def debug_weather_test():
        """날씨 API 테스트 엔드포인트"""
        try:
            latitude = request.args.get("latitude", type=float)
            longitude = request.args.get("longitude", type=float)

            if latitude is None or longitude is None:
                return _invalid(
                    "coordinates",
                    f"lat={latitude}, lng={longitude}",
                    "위도(latitude)와 경도(longitude) 파라미터가 필요합니다.",
                )

            # 좌표 검증
            validated_lat, validated_lng = validate_coordinates(latitude, longitude)

            # 날씨 정보 조회
            weather_info, is_fallback = get_weather_info(validated_lat, validated_lng)

            response_data = {
                "weather": weather_info,
                "is_fallback": is_fallback,
                "coordinates": {"latitude": validated_lat, "longitude": validated_lng},
            }

            return _ok(
                response_data,
                (
                    "날씨 정보 조회 성공"
                    if not is_fallback
                    else "날씨 정보 조회 성공 (폴백 데이터 사용)"
                ),
            )

        except ValueError as ve:
            return _invalid(
                "coordinates",
                f"lat={latitude}, lng={longitude}",
                str(ve),
            )

        except Exception as e:
            logger.exception(f"날씨 테스트 중 오류: {e}")
            return _bad(
                AIErrorCodes.WEATHER_API_FAILED,
                "날씨 API 테스트 중 오류가 발생했습니다.",
                {"error": str(e)},
            )

    @app.route("/api/debug/performance", methods=["GET"])
    def debug_performance():
        """성능 메트릭 조회 엔드포인트"""
        try:
            action = request.args.get("action", "summary")

            if action == "reset":
                performance_monitor.reset_metrics()
                return _ok(
                    {"message": "성능 메트릭이 초기화되었습니다."}, "메트릭 초기화 완료"
                )

            elif action == "summary":
                summary = performance_monitor.get_summary()
                return _ok(
                    {
                        "performance_summary": summary,
                        "total_operations": len(summary),
                    },
                    "성능 메트릭 조회 성공",
                )

            else:
                return _invalid("action", action, "지원되는 action: 'summary', 'reset'")

        except Exception as e:
            logger.exception(f"성능 메트릭 조회 중 오류: {e}")
            return _bad(
                AIErrorCodes.INTERNAL_SERVER_ERROR,
                "성능 메트릭 조회 중 오류가 발생했습니다.",
                {"error": str(e)},
            )
