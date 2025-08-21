# backend/app/routes.py
import os
import json
from flask import request, jsonify, current_app, send_file
from flasgger import swag_from
from datetime import datetime
from werkzeug.utils import secure_filename

from app.schemas import (
    recommend_schema,
    health_check_schema,
    debug_ai_status_schema,
    debug_weather_test_schema,
    serve_image_schema,
)
from app.services import (
    get_weather_info,
    analyze_single_image,
    final_recommendation,
    get_ai_model_status,
)
from app.utils import parse_images, combine_multiple_image_results


def register_routes(app):
    """Flask 라우트를 등록합니다."""

    @app.route("/api/health", methods=["GET"])
    @swag_from(health_check_schema)
    def health_check():
        """서버의 상태를 확인하는 엔드포인트입니다."""
        status = get_ai_model_status()
        return (
            jsonify(
                {
                    "status": "OK",
                    "timestamp": datetime.now().isoformat(),
                    "models_initialized": status["models_initialized"],
                    "yolo_model_loaded": status["yolo_model_loaded"],
                    "resnet_model_loaded": status["resnet_model_loaded"],
                    "db_images_count": len(current_app.config.get("DB_IMAGES", [])),
                }
            ),
            200,
        )

    @app.route("/api/recommend", methods=["POST"])
    @swag_from(recommend_schema)
    def recommend():
        """메인 추천 로직을 처리하는 엔드포인트입니다."""
        # 1. 입력 데이터 파싱 및 유효성 검사 (multipart/form-data 또는 application/json 둘 다 허용)
        data = None
        uploaded_files = []
        if request.content_type and request.content_type.startswith(
            "multipart/form-data"
        ):
            json_data_str = request.form.get("data")
            if not json_data_str:
                return jsonify({"error": "No 'data' field in form"}), 400
            try:
                data = json.loads(json_data_str)
            except json.JSONDecodeError:
                return jsonify({"error": "Invalid JSON format in 'data' field"}), 400
            uploaded_files = parse_images()
        else:
            # JSON 요청 허용 (이미지 없이 동작)
            data = request.get_json(silent=True)
            if not data:
                return (
                    jsonify(
                        {
                            "error": "Unsupported data format. Use multipart/form-data or application/json",
                        }
                    ),
                    400,
                )

        required_fields = ["location", "latitude", "longitude", "user_request"]
        if not all(f in data for f in required_fields):
            return (
                jsonify({"error": f"Missing required fields: {required_fields}"}),
                400,
            )

        page = int(data.get("page", 1))
        per_page = int(data.get("per_page", 3))

        # 2. AI 모델 분석 (이미지가 있는 경우)
        ai_attributes, ai_debug_details = None, []
        if uploaded_files:
            status = get_ai_model_status()
            current_app.logger.info(f"AI 모델 상태: {status}")
            if not status["models_initialized"]:
                current_app.logger.error(
                    "AI 모델이 로드되지 않아 이미지 분석을 건너뜁니다."
                )
                ai_debug_details.append({"error": "AI models are not initialized."})
            else:
                current_app.logger.info(f"업로드된 파일 수: {len(uploaded_files)}")
                all_results = []
                for idx, uploaded_file in enumerate(uploaded_files):
                    current_app.logger.info(
                        f"파일 {idx} 분석 시작: {uploaded_file.filename}"
                    )
                    try:
                        results, debug_info = analyze_single_image(uploaded_file, idx)
                        ai_debug_details.append(debug_info)
                        if results:
                            current_app.logger.info(
                                f"파일 {idx} 분석 성공: {len(results)}개 결과"
                            )
                            all_results.extend(results)
                        else:
                            current_app.logger.warning(f"파일 {idx} 분석 결과 없음")
                    except Exception as e:
                        current_app.logger.error(
                            f"파일 {idx} 분석 중 예외: {e}", exc_info=True
                        )
                        ai_debug_details.append(
                            {"id": idx, "error": f"분석 중 예외: {e}"}
                        )

                if all_results:
                    ai_attributes = combine_multiple_image_results(all_results)
                    current_app.logger.info(f"AI 분석 최종 결과: {ai_attributes}")
                else:
                    current_app.logger.warning("모든 이미지 분석 실패")
        else:
            current_app.logger.info("업로드된 이미지 없음")

        # 3. 스타일 선호도 및 날씨 정보 조회
        style_preferences = data.get("style_select") or (
            ai_attributes.get("style") if ai_attributes else []
        )
        weather_info, used_fallback_weather = get_weather_info(
            data["latitude"], data["longitude"]
        )

        # 4. 최종 추천 생성 (절대로 500 에러가 나지 않도록 완전 안전 처리)
        db_images = current_app.config.get("DB_IMAGES", [])
        current_app.logger.info(f"DB 이미지 개수: {len(db_images)}")

        try:
            recommendation_result = final_recommendation(
                weather=weather_info,
                user_prompt=data["user_request"],
                style_preferences=style_preferences,
                ai_attributes=ai_attributes,
                gemini_api_key=os.getenv("GEMINI_API_KEY"),
                db_images=db_images,
                page=page,
                per_page=per_page,
            )
            current_app.logger.info("final_recommendation 성공")
        except Exception as e:
            current_app.logger.exception(f"final_recommendation 호출 실패: {e}")
            # 완전한 폴백 응답 생성
            temp = weather_info.get("temperature", 20)
            fallback_images = db_images[:per_page] if db_images else []
            
            # 폴백 응답의 이미지 경로도 파일명으로 변환
            converted_fallback_images = []
            for img in fallback_images:
                if isinstance(img, dict) and 'img_path' in img:
                    filename = os.path.basename(img['img_path'])
                    img_copy = img.copy()
                    img_copy['img_path'] = filename
                    converted_fallback_images.append(img_copy)
                else:
                    converted_fallback_images.append(img)
            
            recommendation_result = {
                "recommendation_text": f"오늘 {temp}°C {weather_info.get('condition', '맑음')} 날씨에는 편안한 스타일을 추천합니다.",
                "images": converted_fallback_images,
                "total_pages": (
                    (len(db_images) + per_page - 1) // per_page if db_images else 0
                ),
                "page": 1,
                "suggested_items": style_preferences or ["캐주얼", "편안한"],
            }

        # 5. 안전한 응답 생성 (모든 키가 항상 존재하도록)
        try:
            # 이미지 경로를 파일명만으로 변환 (이미지 서빙 엔드포인트에서 사용하기 위해)
            def convert_image_paths_to_filenames(images_list):
                """이미지 리스트의 img_path를 파일명만으로 변환"""
                if not images_list:
                    return images_list
                
                converted_images = []
                for img in images_list:
                    if isinstance(img, dict) and 'img_path' in img:
                        # 전체 경로에서 파일명만 추출
                        filename = os.path.basename(img['img_path'])
                        img_copy = img.copy()
                        img_copy['img_path'] = filename
                        converted_images.append(img_copy)
                    else:
                        converted_images.append(img)
                return converted_images

            # recommendation_result의 이미지 경로들을 파일명으로 변환
            converted_recommendation_result = recommendation_result.copy()
            if 'images' in converted_recommendation_result:
                converted_recommendation_result['images'] = convert_image_paths_to_filenames(
                    converted_recommendation_result['images']
                )

            # 최종 응답 구조화 (중복 최소화)
            response = {
                "success": True,
                "weather": weather_info or {},
                "styling_tip": converted_recommendation_result.get("recommendation_text", "스타일링 팁을 생성하지 못했습니다."),
                "recommended_images": converted_recommendation_result.get("images", []),
                "suggested_items": converted_recommendation_result.get("suggested_items", []),
                "pagination": {
                    "current_page": converted_recommendation_result.get("page", 1),
                    "total_pages": converted_recommendation_result.get("total_pages", 0),
                },
                # 디버깅 및 상세 정보는 별도 객체로 그룹화 (중복 데이터 제거)
                "debug_info": {
                    "ai_analysis": ai_attributes,
                    "ai_debug_details": ai_debug_details,
                    "weather_fallback_used": used_fallback_weather,
                }
            }
            current_app.logger.info("응답 생성 완료")
            return jsonify(response), 200
        except Exception as e:
            # 최후의 안전장치 - 이것도 실패하면 최소한의 응답
            current_app.logger.exception(f"응답 생성 실패: {e}")
            return (
                jsonify(
                    {
                        "success": True,
                        "weather": weather_info
                        or {"temperature": 20, "condition": "맑음"},
                        "recommendation_text": "기본 추천을 생성했습니다.",
                        "ai_analysis": ai_attributes,
                        "ai_debug_details": ai_debug_details,
                        "recommended_images": [],
                        "total_pages": 0,
                        "current_page": 1,
                    }
                ),
                200,
            )

    @app.route("/api/debug/ai_status", methods=["GET"])
    @app.route("/api/debug/ai-status", methods=["GET"])  # 하이픈 버전도 지원
    @swag_from(debug_ai_status_schema)
    def debug_ai_status():
        """AI 모델의 상태를 확인하는 디버그용 엔드포인트입니다."""
        return jsonify(get_ai_model_status()), 200

    @app.route("/api/test/predict", methods=["POST"])
    def test_predict():
        """이미지만으로 예측 테스트 - 간단한 예측값 확인용"""
        try:
            uploaded_files = parse_images()
            if not uploaded_files:
                return jsonify({"error": "이미지가 필요합니다"}), 400

            status = get_ai_model_status()
            if not status["models_initialized"]:
                return jsonify({"error": "AI 모델이 초기화되지 않았습니다"}), 500

            file = uploaded_files[0]
            results, debug_info = analyze_single_image(file, 0)

            if results:
                ai_attributes = combine_multiple_image_results(results)
                return (
                    jsonify(
                        {
                            "success": True,
                            "filename": file.filename,
                            "ai_analysis": ai_attributes,
                            "raw_results": results,
                            "debug_info": debug_info,
                        }
                    ),
                    200,
                )
            else:
                return (
                    jsonify(
                        {
                            "success": False,
                            "error": "분석 실패",
                            "debug_info": debug_info,
                        }
                    ),
                    200,
                )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/debug/weather_test", methods=["GET"])
    @swag_from(debug_weather_test_schema)
    def debug_weather_test():
        """날씨 API를 테스트하는 디버그용 엔드포인트입니다."""
        lat = request.args.get("lat", "37.5665")
        lon = request.args.get("lon", "126.9780")
        try:
            lat, lon = float(lat), float(lon)
            weather_info, used_fallback = get_weather_info(lat, lon)
            return (
                jsonify(
                    {
                        "latitude": lat,
                        "longitude": lon,
                        "weather_info": weather_info,
                        "used_fallback": used_fallback,
                        "used_fallback": used_fallback,
                    }
                ),
                200,
            )
        except ValueError:
            return jsonify({"error": "Invalid lat/lon parameters"}), 400
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/images/<path:filename>", methods=["GET"])
    @swag_from(serve_image_schema)
    def serve_image(filename):
        try:
            image_dir = current_app.config.get("IMAGE_DIR")
            if not image_dir:
                current_app.logger.error("IMAGE_DIR is not configured.")
                return jsonify({"error": "Image directory not configured"}), 500

            # 1. 파일명을 안전하게 정제 (경로 문자 제거)
            safe_filename = secure_filename(filename)
            if not safe_filename:
                return jsonify({"error": "Invalid filename"}), 400

            # 2. 절대 경로 생성 및 검증
            image_dir_abs = os.path.abspath(image_dir)
            file_path_abs = os.path.abspath(os.path.join(image_dir_abs, safe_filename))

            # 최종 경로가 이미지 디렉토리 내부에 있는지 확인 (가장 중요한 보안)
            if not file_path_abs.startswith(image_dir_abs):
                current_app.logger.warning(f"Path traversal attempt denied: {filename}")
                return jsonify({"error": "Access denied"}), 403
            
            # 파일 존재 여부 확인
            if not os.path.exists(file_path_abs):
                return jsonify({"error": "Image not found"}), 404
            
            # 파일 서빙
            return send_file(file_path_abs, as_attachment=False)
        except Exception as e:
            current_app.logger.error(f"이미지 서빙 실패: {filename}, 오류: {e}", exc_info=True)
            return jsonify({"error": "Internal server error"}), 500

    @app.route("/api/makeswagger", methods=["GET"])
    def make_swagger():
        """현재 API 스펙을 기반으로 swagger.yaml 파일을 backend 폴더에 생성합니다."""
        try:
            # Swagger 2.0 YAML 내용 (사용자 제공 스펙을 최대한 반영, 확인용 문서)
            swagger_yaml = """
swagger: '2.0'
info:
    title: OOTD-AI API DOCS
    description: AI 기반 의상 추천 서비스 API (확인용 Swagger 문서)
    version: 1.0.0
host: localhost:5000
basePath: /
schemes:
- http
consumes:
- application/json
- multipart/form-data
produces:
- application/json
definitions:
    Error:
        type: object
        properties:
            error:
                type: string
                description: 오류 메시지
        required:
        - error
paths:
    /api/recommend:
        post:
            tags:
            - Recommendation
            summary: AI 기반 의상 추천
            description: JSON 데이터와 이미지를 함께 받아서 AI가 의상을 추천합니다
            consumes:
            - multipart/form-data
            produces:
            - application/json
            parameters:
            - name: data
                in: formData
                type: string
                required: true
                description: 'JSON 형태의 요청 데이터 (예: {"location": "서울", "latitude": 37.5665, "longitude": 126.9780, "style_select": ["스트릿","캐주얼"], "user_request": "귀엽게 입고 싶어요"})'
            - name: image
                in: formData
                type: file
                required: false
                description: 의류 이미지 파일 (선택사항)
            responses:
                200:
                    description: 추천 성공
                    schema:
                        type: object
                        properties:
                            success:
                                type: boolean
                            weather:
                                type: object
                                properties:
                                    temperature:
                                        type: number
                                    condition:
                                        type: string
                                    humidity:
                                        type: integer
                                    wind_speed:
                                        type: number
                            recommendation_text:
                                type: string
                            recommended_images:
                                type: array
                                items:
                                    type: object
                                    properties:
                                        img_path:
                                            type: string
                                        similarity_score:
                                            type: number
                                        label:
                                            type: object
                            image_analysis:
                                type: object
                                description: 업로드된 이미지 분석 결과 (디버그용)
                400:
                    description: 잘못된 요청
                    schema:
                        $ref: '#/definitions/Error'
                500:
                    description: 서버 오류
                    schema:
                        $ref: '#/definitions/Error'
    /api/health:
        get:
            tags:
            - Utility
            summary: 서버 상태 확인
            description: 서버가 정상적으로 동작하는지 확인합니다
            produces:
            - application/json
            responses:
                200:
                    description: 서버 정상
                    schema:
                        type: object
                        properties:
                            status:
                                type: string
                            timestamp:
                                type: string
                            models_initialized:
                                type: boolean
                            yolo_model_loaded:
                                type: boolean
                            resnet_model_loaded:
                                type: boolean
                            db_images_count:
                                type: integer
    /api/makeswagger:
        get:
            tags:
            - Utility
            summary: Swagger YAML 파일 생성
            description: 현재 API 스펙을 바탕으로 swagger.yaml 파일을 서버 폴더에 생성합니다(github/online 에디터 공유용)
            produces:
            - application/json
            responses:
                200:
                    description: Swagger 파일 생성 성공
                    schema:
                        type: object
                        properties:
                            message:
                                type: string
                            file_path:
                                type: string
                            timestamp:
                                type: string
                            file_size:
                                type: string
                500:
                    description: Swagger 파일 생성 실패
                    schema:
                        $ref: '#/definitions/Error'
    /api/images/{filename}:
        get:
            tags:
            - Utility
            summary: 이미지 파일 서빙
            description: 백엔드 서버의 이미지 파일을 클라이언트에게 제공합니다
            produces:
            - image/jpeg
            - image/png
            - image/gif
            - image/webp
            - application/json
            parameters:
            - name: filename
                in: path
                type: string
                required: true
                description: 이미지 파일명 (예⁚ 1084011.jpg)
            responses:
                200:
                    description: 이미지 파일 반환
                    schema:
                        type: file
                400:
                    description: 잘못된 파일명
                    schema:
                        $ref: '#/definitions/Error'
                403:
                    description: 접근 권한 없음
                    schema:
                        $ref: '#/definitions/Error'
                404:
                    description: 파일을 찾을 수 없음
                    schema:
                        $ref: '#/definitions/Error'
                500:
                    description: 서버 오류
                    schema:
                        $ref: '#/definitions/Error'
"""

            backend_dir = os.path.dirname(current_app.root_path)
            file_path = os.path.join(backend_dir, "swagger.yaml")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(swagger_yaml.strip() + "\n")

            stat = os.stat(file_path)
            return (
                jsonify(
                    {
                        "message": "swagger.yaml generated",
                        "file_path": os.path.abspath(file_path),
                        "timestamp": datetime.now().isoformat(),
                        "file_size": f"{stat.st_size} bytes",
                    }
                ),
                200,
            )
        except Exception as e:
            current_app.logger.error(f"Swagger 파일 생성 실패: {e}", exc_info=True)
            return jsonify({"error": str(e)}), 500
