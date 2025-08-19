"""
백엔드 API를 위한 Swagger 스키마 정의
구조화되고 재사용 가능한 스키마 컴포넌트들을 포함
"""

from typing import Dict, Any


class SwaggerSchemas:
    """Swagger 스키마 컴포넌트 클래스"""

    # 공통 응답 스키마
    @staticmethod
    def get_common_error_schema() -> Dict[str, Any]:
        """공통 에러 응답 스키마"""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": False},
                "error_code": {"type": "string", "example": "AI-001"},
                "message": {"type": "string", "example": "오류 메시지"},
                "timestamp": {"type": "string", "format": "date-time"},
                "details": {"type": "object", "description": "추가 오류 정보"},
                "error_info": {
                    "type": "object",
                    "properties": {
                        "category": {"type": "string"},
                        "severity": {"type": "string"},
                        "description": {"type": "string"},
                        "suggestion": {"type": "string"},
                    },
                },
            },
            "required": ["success", "error_code", "message", "timestamp"],
        }

    @staticmethod
    def get_common_success_schema() -> Dict[str, Any]:
        """공통 성공 응답 스키마"""
        return {
            "type": "object",
            "properties": {
                "success": {"type": "boolean", "example": True},
                "message": {"type": "string", "example": "Success"},
                "timestamp": {"type": "string", "format": "date-time"},
                "data": {"type": "object", "description": "응답 데이터"},
                "metadata": {"type": "object", "description": "메타데이터"},
            },
            "required": ["success", "message", "timestamp", "data"],
        }

    # 도메인별 스키마
    @staticmethod
    def get_weather_schema() -> Dict[str, Any]:
        """날씨 정보 스키마"""
        return {
            "type": "object",
            "properties": {
                "temperature": {
                    "type": "number",
                    "example": 23.5,
                    "description": "온도 (섭씨)",
                },
                "condition": {
                    "type": "string",
                    "example": "맑음",
                    "description": "날씨 상태",
                },
                "humidity": {
                    "type": "integer",
                    "example": 60,
                    "description": "습도 (%)",
                },
                "wind_speed": {
                    "type": "number",
                    "example": 5.0,
                    "description": "풍속 (m/s)",
                },
                "error_code": {
                    "type": "string",
                    "description": "날씨 API 에러 코드 (선택적)",
                },
            },
            "required": ["temperature", "condition", "humidity", "wind_speed"],
        }

    @staticmethod
    def get_ai_analysis_schema() -> Dict[str, Any]:
        """AI 분석 결과 스키마"""
        return {
            "type": "object",
            "properties": {
                "yolo_results": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer", "description": "파일 인덱스"},
                            "filename": {"type": "string", "description": "파일명"},
                            "object_index": {
                                "type": "integer",
                                "description": "객체 인덱스",
                            },
                            "category": {
                                "type": "string",
                                "description": "탐지된 카테고리",
                            },
                            "confidence": {"type": "number", "description": "신뢰도"},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "number"},
                                "description": "바운딩 박스 좌표 [x1, y1, x2, y2]",
                            },
                            "attributes": {
                                "type": "object",
                                "description": "ResNet 분석 속성",
                                "properties": {
                                    "color": {"type": "string"},
                                    "style": {"type": "string"},
                                    "material": {"type": "string"},
                                },
                            },
                        },
                    },
                },
                "combined_analysis": {
                    "type": "object",
                    "properties": {
                        "combined_categories": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "combined_colors": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "combined_styles": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "confidence_summary": {
                            "type": "object",
                            "properties": {
                                "avg": {"type": "number"},
                                "min": {"type": "number"},
                                "max": {"type": "number"},
                            },
                        },
                        "image_count": {"type": "integer"},
                    },
                },
            },
        }

    @staticmethod
    def get_model_status_schema() -> Dict[str, Any]:
        """모델 상태 스키마"""
        return {
            "type": "object",
            "properties": {
                "models_initialized": {"type": "boolean"},
                "yolo_model_loaded": {"type": "boolean"},
                "resnet_model_loaded": {"type": "boolean"},
                "device": {"type": "string", "example": "cuda"},
                "modules_available": {
                    "type": "object",
                    "properties": {
                        "YOLOv11MultiTask": {"type": "boolean"},
                        "FashionAttributePredictor": {"type": "boolean"},
                        "KoreaWeatherAPI": {"type": "boolean"},
                        "final_recommendation": {"type": "boolean"},
                    },
                },
                "model_instances": {
                    "type": "object",
                    "properties": {
                        "yolo": {"type": "string"},
                        "resnet": {"type": "string"},
                        "weather": {"type": "string"},
                    },
                },
                "model_files": {
                    "type": "object",
                    "properties": {
                        "yolo": {
                            "type": "object",
                            "properties": {
                                "exists": {"type": "boolean"},
                                "path": {"type": "string"},
                                "size_mb": {"type": "number"},
                            },
                        },
                        "resnet": {
                            "type": "object",
                            "properties": {
                                "exists": {"type": "boolean"},
                                "path": {"type": "string"},
                                "size_mb": {"type": "number"},
                            },
                        },
                    },
                },
                "model_load_time": {"type": "string", "format": "date-time"},
                "total_model_size_mb": {"type": "number"},
            },
        }


# 개별 API 스키마 정의
recommend_schema = {
    "tags": ["Recommendation"],
    "summary": "AI 기반 의상 추천",
    "description": """
    JSON 데이터와 이미지를 함께 받아서 AI가 의상을 추천합니다.
    
    요청 데이터 예시:
    ```json
    {
        "location": "서울",
        "latitude": 37.5665,
        "longitude": 126.9780,
        "style_select": ["스트릿", "캐주얼"],
        "user_request": "귀엽게 입고 싶어요"
    }
    ```
    """,
    "consumes": ["multipart/form-data"],
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "data",
            "in": "formData",
            "type": "string",
            "required": True,
            "description": "JSON 형태의 요청 데이터",
            "example": '{"location": "서울", "latitude": 37.5665, "longitude": 126.9780, "style_select": ["스트릿", "캐주얼"], "user_request": "귀엽게 입고 싶어요"}',
        },
        {
            "name": "images",
            "in": "formData",
            "type": "file",
            "required": False,
            "description": "의류 이미지 파일들 (최대 5개, 각각 최대 5MB)",
            "allowMultiple": True,
        },
    ],
    "responses": {
        200: {
            "description": "추천 성공",
            "schema": {
                "allOf": [
                    SwaggerSchemas.get_common_success_schema(),
                    {
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "weather": SwaggerSchemas.get_weather_schema(),
                                    "recommendation_text": {
                                        "type": "string",
                                        "description": "AI 추천 텍스트",
                                    },
                                    "suggested_items": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                        "description": "추천 아이템 목록",
                                    },
                                    "ai_analysis": SwaggerSchemas.get_ai_analysis_schema(),
                                    "recommendation_details": {
                                        "type": "object",
                                        "description": "세부 추천 결과",
                                    },
                                    "processing_info": {
                                        "type": "object",
                                        "properties": {
                                            "total_processing_time_ms": {
                                                "type": "number"
                                            },
                                            "image_count": {"type": "integer"},
                                            "weather_source": {
                                                "type": "string",
                                                "enum": ["api", "fallback"],
                                            },
                                        },
                                    },
                                },
                            }
                        }
                    },
                ]
            },
        },
        400: {
            "description": "잘못된 요청 (입력 검증 실패)",
            "schema": SwaggerSchemas.get_common_error_schema(),
        },
        500: {
            "description": "서버 내부 오류",
            "schema": SwaggerSchemas.get_common_error_schema(),
        },
    },
}


health_check_schema = {
    "tags": ["System"],
    "summary": "서버 상태 확인",
    "description": "서버와 AI 모델의 상태를 확인합니다",
    "produces": ["application/json"],
    "responses": {
        200: {
            "description": "서버 상태 정보",
            "schema": {
                "allOf": [
                    SwaggerSchemas.get_common_success_schema(),
                    {
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "status": {
                                        "type": "string",
                                        "enum": ["OK", "WARNING", "ERROR"],
                                    },
                                    "model_details": SwaggerSchemas.get_model_status_schema(),
                                    "system": {
                                        "type": "object",
                                        "properties": {
                                            "gpu_available": {"type": "boolean"},
                                            "gpu_count": {"type": "integer"},
                                            "gpu_memory": {
                                                "type": "object",
                                                "properties": {
                                                    "allocated_mb": {"type": "number"},
                                                    "reserved_mb": {"type": "number"},
                                                },
                                            },
                                        },
                                    },
                                    "performance": {
                                        "type": "object",
                                        "properties": {
                                            "metrics": {
                                                "type": "object",
                                                "description": "성능 메트릭",
                                            },
                                            "slo_violations": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "operation": {"type": "string"},
                                                        "actual_ms": {"type": "number"},
                                                        "limit_ms": {"type": "number"},
                                                        "violation_percent": {
                                                            "type": "number"
                                                        },
                                                    },
                                                },
                                            },
                                        },
                                    },
                                    "db_images_count": {
                                        "type": "integer",
                                        "description": "로드된 DB 이미지 수",
                                    },
                                },
                            }
                        }
                    },
                ]
            },
        }
    },
}


debug_ai_status_schema = {
    "tags": ["Debug"],
    "summary": "AI 모델 상태 디버그",
    "description": "AI 모델의 상세한 상태 정보를 반환합니다",
    "produces": ["application/json"],
    "responses": {
        200: {
            "description": "AI 모델 상태",
            "schema": {
                "allOf": [
                    SwaggerSchemas.get_common_success_schema(),
                    {"properties": {"data": SwaggerSchemas.get_model_status_schema()}},
                ]
            },
        }
    },
}


debug_weather_test_schema = {
    "tags": ["Debug"],
    "summary": "날씨 API 테스트",
    "description": "특정 좌표의 날씨 정보를 테스트합니다",
    "produces": ["application/json"],
    "parameters": [
        {
            "name": "latitude",
            "in": "query",
            "type": "number",
            "required": True,
            "description": "위도 (33.0 ~ 38.9)",
            "minimum": 33.0,
            "maximum": 38.9,
        },
        {
            "name": "longitude",
            "in": "query",
            "type": "number",
            "required": True,
            "description": "경도 (124.0 ~ 132.0)",
            "minimum": 124.0,
            "maximum": 132.0,
        },
    ],
    "responses": {
        200: {
            "description": "날씨 정보",
            "schema": {
                "allOf": [
                    SwaggerSchemas.get_common_success_schema(),
                    {
                        "properties": {
                            "data": {
                                "type": "object",
                                "properties": {
                                    "weather": SwaggerSchemas.get_weather_schema(),
                                    "is_fallback": {
                                        "type": "boolean",
                                        "description": "폴백 데이터 사용 여부",
                                    },
                                    "coordinates": {
                                        "type": "object",
                                        "properties": {
                                            "latitude": {"type": "number"},
                                            "longitude": {"type": "number"},
                                        },
                                    },
                                },
                            }
                        }
                    },
                ]
            },
        },
        400: {
            "description": "잘못된 좌표",
            "schema": SwaggerSchemas.get_common_error_schema(),
        },
    },
}


serve_image_schema = {
    "tags": ["Utility"],
    "summary": "이미지 파일 제공",
    "description": "서버에 저장된 이미지 파일을 제공합니다",
    "produces": ["image/jpeg", "image/png"],
    "parameters": [
        {
            "name": "filename",
            "in": "path",
            "type": "string",
            "required": True,
            "description": "이미지 파일명",
            "example": "1002248.jpg",
        },
    ],
    "responses": {
        200: {"description": "이미지 파일", "schema": {"type": "file"}},
        404: {
            "description": "파일을 찾을 수 없음",
            "schema": SwaggerSchemas.get_common_error_schema(),
        },
    },
}
