"""
백엔드 API를 위한 Swagger 2.0 스키마.
명확성, 일관성 및 리팩터링된 애플리케이션 로직과의 정렬을 위해 최적화됨.
"""

# 추천 API 스키마 (Swagger 2.0 형식)
recommend_schema = {
    "tags": ["Recommendation"],
    "summary": "AI 기반 의상 추천",
    "description": "JSON 데이터와 이미지를 함께 받아서 AI가 의상을 추천합니다",
    "consumes": ["multipart/form-data"],
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "data",
            "in": "formData",
            "type": "string",
            "required": True,
            "description": 'JSON 형태의 요청 데이터 (예: {"location": "서울", "latitude": 37.5665, "longitude": 126.9780, "style_select": ["스트릿", "캐주얼"], "user_request": "귀엽게 입고 싶어요"})',
        },
        {
            "name": "images",
            "in": "formData",
            "type": "file",
            "required": False,
            "description": "의류 이미지 파일들 (여러 개 업로드 가능)",
            "allowMultiple": True,
        },
    ],
    "responses": {
        200: {
            "description": "추천 성공",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean"},
                    "weather": {
                        "type": "object",
                        "properties": {
                            "temperature": {"type": "number"},
                            "condition": {"type": "string"},
                            "humidity": {"type": "integer"},
                            "wind_speed": {"type": "number"},
                        },
                    },
                    "recommendation_text": {"type": "string"},
                    "suggested_items": {"type": "array", "items": {"type": "string"}},
                    "ai_analysis": {
                        "type": "object",
                        "description": "업로드된 이미지 AI 분석 결과",
                    },
                    "recommendation_details": {
                        "type": "object",
                        "description": "세부 추천 결과",
                    },
                },
            },
        },
        400: {
            "description": "잘못된 요청",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "오류 메시지"}
                },
            },
        },
        500: {
            "description": "서버 오류",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "오류 메시지"}
                },
            },
        },
    },
}

# 헬스 체크 API 스키마
health_check_schema = {
    "tags": ["Utility"],
    "summary": "서버 상태 확인",
    "description": "서버가 정상적으로 동작하는지 확인합니다",
    "produces": ["application/json"],
    "responses": {
        200: {
            "description": "서버 정상",
            "schema": {
                "type": "object",
                "properties": {
                    "status": {"type": "string", "description": "서버 상태"},
                    "timestamp": {
                        "type": "string",
                        "description": "ISO 형식 타임스탬프",
                    },
                    "models_initialized": {
                        "type": "boolean",
                        "description": "모델 초기화 완료 상태",
                    },
                    "yolo_model_loaded": {
                        "type": "boolean",
                        "description": "YOLO 모델 로드 상태",
                    },
                    "resnet_model_loaded": {
                        "type": "boolean",
                        "description": "ResNet 모델 로드 상태",
                    },
                    "db_images_count": {
                        "type": "integer",
                        "description": "데이터베이스에서 로드된 이미지 수",
                    },
                },
            },
        }
    },
}

# Swagger 파일 생성 API 스키마
make_swagger_schema = {
    "tags": ["Utility"],
    "summary": "Swagger YAML 파일 생성",
    "description": "현재 API 스펙을 바탕으로 swagger.yaml 파일을 서버 폴더에 생성합니다",
    "produces": ["application/json"],
    "responses": {
        200: {
            "description": "Swagger 파일 생성 성공",
            "schema": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "성공 메시지"},
                    "file_path": {
                        "type": "string",
                        "description": "생성된 파일의 절대 경로",
                    },
                    "timestamp": {"type": "string", "description": "파일 생성 시간"},
                },
            },
        },
        500: {
            "description": "Swagger 파일 생성 실패",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "오류 메시지"}
                },
            },
        },
    },
}

# AI 상태 디버그 스키마
debug_ai_status_schema = {
    "tags": ["Debug"],
    "summary": "AI 모델 상태 확인",
    "description": "AI 모델의 로드 상태와 초기화 상태를 확인합니다",
    "produces": ["application/json"],
    "responses": {
        200: {
            "description": "AI 상태 조회 성공",
            "schema": {
                "type": "object",
                "properties": {
                    "models_initialized": {
                        "type": "boolean",
                        "description": "모델 초기화 상태",
                    },
                    "yolo_model_loaded": {
                        "type": "boolean",
                        "description": "YOLO 모델 로드 상태",
                    },
                    "resnet_model_loaded": {
                        "type": "boolean",
                        "description": "ResNet 모델 로드 상태",
                    },
                    "yolo_model_instance": {
                        "type": "string",
                        "description": "YOLO 모델 인스턴스 이름",
                    },
                    "resnet_model_instance": {
                        "type": "string",
                        "description": "ResNet 모델 인스턴스 이름",
                    },
                    "YOLOv11MultiTask_available": {
                        "type": "boolean",
                        "description": "YOLOv11MultiTask 클래스 사용 가능 여부",
                    },
                    "model_file_exists": {
                        "type": "boolean",
                        "description": "모델 파일 존재 여부",
                    },
                },
            },
        }
    },
}

# 날씨 API 테스트 스키마
debug_weather_test_schema = {
    "tags": ["Debug"],
    "summary": "날씨 API 테스트",
    "description": "실제 한국 기상청 API를 호출하여 날씨 정보를 조회합니다",
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "lat",
            "in": "query",
            "type": "number",
            "required": False,
            "default": 37.5665,
            "description": "위도 (기본값: 서울 37.5665)",
        },
        {
            "name": "lon",
            "in": "query",
            "type": "number",
            "required": False,
            "default": 126.9780,
            "description": "경도 (기본값: 서울 126.9780)",
        },
    ],
    "responses": {
        200: {
            "description": "날씨 API 테스트 성공",
            "schema": {
                "type": "object",
                "properties": {
                    "success": {"type": "boolean", "description": "API 호출 성공 여부"},
                    "latitude": {"type": "number", "description": "요청한 위도"},
                    "longitude": {"type": "number", "description": "요청한 경도"},
                    "weather_info": {
                        "type": "object",
                        "properties": {
                            "temperature": {
                                "type": "number",
                                "description": "기온 (°C)",
                            },
                            "condition": {"type": "string", "description": "날씨 상태"},
                            "humidity": {"type": "integer", "description": "습도 (%)"},
                            "wind_speed": {
                                "type": "number",
                                "description": "풍속 (m/s)",
                            },
                        },
                        "description": "날씨 정보",
                    },
                    "used_fallback": {
                        "type": "boolean",
                        "description": "폴백 데이터 사용 여부",
                    },
                },
            },
        },
        400: {
            "description": "잘못된 위도/경도 매개변수",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "오류 메시지"}
                },
            },
        },
        500: {
            "description": "날씨 API 호출 실패",
            "schema": {
                "type": "object",
                "properties": {
                    "error": {"type": "string", "description": "오류 메시지"}
                },
            },
        },
    },
}

# 기존 코드와의 호환성을 위한 별칭들
RECOMMENDED_IMAGE_SCHEMA = recommend_schema["responses"][200]["schema"]["properties"][
    "ai_analysis"
]
WEATHER_SCHEMA = recommend_schema["responses"][200]["schema"]["properties"]["weather"]
DEBUG_INFO_SCHEMA = {"type": "object", "description": "디버그 정보"}
# recommend_schema는 이미 정의됨
