# backend/app/routes/__init__.py
"""
라우트 모듈 패키지
API 엔드포인트들을 기능별로 분리하여 관리
"""

from flask import Flask
from .health import register_health_routes
from .recommendation import register_recommendation_routes
from .debug import register_debug_routes
from .utility import register_utility_routes


def register_routes(app: Flask):
    """모든 라우트를 Flask 앱에 등록합니다."""
    register_health_routes(app)
    register_recommendation_routes(app)
    register_debug_routes(app)
    register_utility_routes(app)
