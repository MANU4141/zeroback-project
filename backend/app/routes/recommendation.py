# backend/app/routes/recommendation.py
"""
추천 관련 라우트
"""
# Standard library imports
import json
import logging

# Third-party imports
from flask import request, jsonify, current_app
from flasgger import swag_from

# Local imports
from app.schemas import recommend_schema
from app.services import (
    get_weather_info,
    analyze_single_image,
    get_final_recommendation,
)
from app.error_codes import AIErrorCodes, create_error_response
from app.utils import (
    parse_images,
    combine_multiple_image_results,
    validate_coordinates,
    convert_image_paths_to_filenames,
)
from app.monitoring import performance_timer, create_timing_context

logger = logging.getLogger(__name__)


def register_recommendation_routes(app):
    """추천 관련 라우트 등록"""

    @app.route("/api/recommend", methods=["POST"])
    @swag_from(recommend_schema)
    def recommend():
        """AI 기반 의상 추천 엔드포인트"""
        timing_context = create_timing_context()

        try:
            # 1. 요청 데이터 파싱 및 검증
            with performance_timer("request_parsing"):
                request_data = _parse_request_data()
                timing_context.record_step("request_parsing")

            # 2. 좌표 검증
            latitude, longitude = validate_coordinates(
                request_data["latitude"], request_data["longitude"]
            )

            # 3. 이미지 처리 (선택적)
            ai_analysis_result = None
            uploaded_files = parse_images()

            if uploaded_files:
                with performance_timer("image_analysis"):
                    ai_analysis_result = _process_uploaded_images(uploaded_files)
                    timing_context.record_step("image_analysis")

            # 4. 날씨 정보 조회
            with performance_timer("weather_api"):
                weather_info, is_weather_fallback = get_weather_info(
                    latitude, longitude
                )
                timing_context.record_step("weather_api")

            # 5. 최종 추천 생성
            with performance_timer("recommendation_generation"):
                recommendation_result = _generate_recommendation(
                    request_data, weather_info, ai_analysis_result
                )
                timing_context.record_step("recommendation_generation")

            # 6. 응답 생성
            total_timings = timing_context.finalize("total_request")

            response_data = {
                "weather": weather_info,
                "recommendation_text": recommendation_result.get(
                    "recommendation_text", ""
                ),
                "suggested_items": recommendation_result.get("suggested_items", []),
                "ai_analysis": ai_analysis_result,
                "recommendation_details": recommendation_result.get("details", {}),
                "processing_info": {
                    "total_processing_time_ms": total_timings.get("total_request", 0),
                    "image_count": len(uploaded_files) if uploaded_files else 0,
                    "weather_source": "fallback" if is_weather_fallback else "api",
                    "timings": total_timings,
                },
            }

            logger.info(
                f"추천 완료: 이미지 {len(uploaded_files) if uploaded_files else 0}개, "
                f"총 처리시간 {total_timings.get('total_request', 0):.2f}ms"
            )

            return jsonify(
                {
                    "success": True,
                    "message": "추천이 성공적으로 완료되었습니다.",
                    "data": response_data,
                }
            )

        except ValueError as ve:
            logger.warning(f"입력 검증 실패: {ve}")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": "VALIDATION_ERROR",
                        "message": "입력 데이터가 유효하지 않습니다.",
                        "details": {"field": "request_data", "reason": str(ve)},
                    }
                ),
                400,
            )

        except Exception as e:
            logger.exception("추천 처리 중 오류")
            return (
                jsonify(
                    {
                        "success": False,
                        "error_code": AIErrorCodes.INTERNAL_SERVER_ERROR,
                        "message": "추천 처리 중 예상치 못한 오류가 발생했습니다.",
                        "details": {"error": str(e)},
                    }
                ),
                500,
            )


def _parse_request_data():
    """요청 데이터를 파싱하고 검증합니다."""
    # JSON 바디 먼저 확인
    data_json = request.get_json(silent=True)
    if data_json and isinstance(data_json, dict):
        data = data_json
    else:
        # form-data에서 'data' 필드 확인
        raw_data = request.form.get("data")
        if not raw_data:
            raise ValueError(
                "'data' 필드가 필요합니다. JSON 바디 또는 form-data 'data'로 전달하세요."
            )

        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 파싱 실패: {e}")

    # 필수 필드 검증
    required_fields = ["latitude", "longitude"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"필수 필드 '{field}'가 누락되었습니다.")

    # 기본값 설정
    data.setdefault("location", "알 수 없는 위치")
    data.setdefault("style_select", [])
    data.setdefault("user_request", "")

    return data


def _process_uploaded_images(uploaded_files):
    """업로드된 이미지들을 처리합니다."""
    if not uploaded_files:
        return None

    image_results = []
    debug_infos = []

    for idx, uploaded_file in enumerate(uploaded_files):
        try:
            analysis_result, debug_info = analyze_single_image(uploaded_file, idx)

            if analysis_result:
                image_results.extend(analysis_result)
            debug_infos.append(debug_info)

        except Exception as e:
            logger.exception(f"이미지 {idx} 분석 실패")
            debug_infos.append(
                {
                    "id": idx,
                    "filename": uploaded_file.filename,
                    "error": str(e),
                    "error_code": AIErrorCodes.IMAGE_PROCESSING_FAILED,
                }
            )

    # 결과 통합
    combined_results = combine_multiple_image_results(image_results)

    return {
        "yolo_results": image_results,
        "combined_analysis": combined_results,
        "processing_debug": debug_infos,
        "summary": {
            "total_images": len(uploaded_files),
            "successful_analyses": len(image_results),
            "failed_analyses": len([d for d in debug_infos if "error" in d]),
        },
    }


def _generate_recommendation(request_data, weather_info, ai_analysis_result):
    """최종 추천을 생성합니다."""
    try:
        # DB 이미지 데이터 가져오기
        db_images = current_app.config.get("DB_IMAGES", [])
        converted_db_images = convert_image_paths_to_filenames(db_images)

        # 추천 생성
        recommendation_result = get_final_recommendation(
            weather=weather_info,
            user_prompt=request_data.get("user_request", "편안한 옷차림"),
            style_preferences=request_data.get("style_select", ["캐주얼"]),
            ai_attributes=ai_analysis_result,
            db_images=converted_db_images,
        )

        return recommendation_result

    except Exception as e:
        logger.exception("추천 생성 실패")
        return {
            "recommendation_text": "추천을 생성할 수 없습니다. 기본 추천을 제공합니다.",
            "suggested_items": ["기본 의상 추천"],
            "details": {"error": str(e)},
            "error_code": AIErrorCodes.LLM_API_FAILED,
        }
