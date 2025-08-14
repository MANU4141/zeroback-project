# backend/app/__init__.py
import os
import logging
from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

# 내부 모듈 import
from config import Config
from app.services import initialize_ai_models
from app.utils import build_db_images as _build_db_images_internal
from app.routes import register_routes


def create_app():
    """Flask 애플리케이션을 생성하고 설정합니다."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
    )

    # CORS 설정
    CORS(app, resources={r"/api/*": {"origins": os.getenv("CORS_ORIGINS", "*")}})

    # Swagger 설정
    Swagger(app)

    # 애플리케이션 컨텍스트 내에서 초기화 진행
    with app.app_context():
        # DB 이미지 빌드
        app.config["DB_IMAGES"] = _build_db_images_internal(
            app.config["LABELS_DIR"], app.config["IMAGE_DIR"]
        )

        # AI 모델 초기화
        initialize_ai_models(app)

    # 라우트 등록
    register_routes(app)

    return app


# 테스트 및 외부 호환을 위한 헬퍼 래퍼 함수 노출
def build_db_images(app_or_labels_dir, image_dir=None):
    """DB 이미지를 빌드하는 호환 래퍼.

    - build_db_images(app) 형태를 지원 (app.config의 LABELS_DIR/IMAGE_DIR 사용)
    - build_db_images(labels_dir, image_dir) 형태도 지원
    """
    if image_dir is None and hasattr(app_or_labels_dir, "config"):
        labels_dir = app_or_labels_dir.config.get("LABELS_DIR")
        images_dir = app_or_labels_dir.config.get("IMAGE_DIR")
        return _build_db_images_internal(labels_dir, images_dir)
    elif image_dir is not None:
        return _build_db_images_internal(app_or_labels_dir, image_dir)
    else:
        raise TypeError(
            "build_db_images requires either (app) or (labels_dir, image_dir) arguments"
        )
