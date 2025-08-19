# backend/app/__init__.py
# Standard library imports
import logging
import os

# Third-party imports
from flask import Flask
from flask_cors import CORS
from flasgger import Swagger

# Local imports
from config import Config
from app.services import initialize_ai_models
from app.utils import build_db_images
from app.routes import register_routes

logger = logging.getLogger(__name__)


def create_app():
    """Flask 애플리케이션을 생성하고 설정합니다."""
    app = Flask(__name__)
    app.config.from_object(Config)

    # 로깅 설정
    setup_logging()

    # CORS 설정
    setup_cors(app)

    # Swagger 설정
    setup_swagger(app)

    # 애플리케이션 컨텍스트 내에서 초기화 진행
    with app.app_context():
        try:
            # DB 이미지 빌드
            logger.info("이미지 데이터베이스 구축 중...")
            app.config["DB_IMAGES"] = build_db_images(
                app.config["LABELS_DIR"], app.config["IMAGE_DIR"]
            )
            logger.info(
                f"이미지 데이터베이스 구축 완료: {len(app.config['DB_IMAGES'])}개 이미지"
            )

            # AI 모델 초기화
            logger.info("AI 모델 초기화 중...")
            initialize_ai_models(app)
            logger.info("AI 모델 초기화 완료")

        except Exception as e:
            logger.error(f"애플리케이션 초기화 중 오류 발생: {e}", exc_info=True)
            # 오류가 발생해도 서버는 시작하되, 상태를 기록
            app.config["INITIALIZATION_ERROR"] = str(e)

    # 라우트 등록
    register_routes(app)

    return app


def setup_logging():
    """로깅을 설정합니다."""
    # 환경변수에서 로그 레벨 가져오기
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()

    # 간단한 로깅 설정
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="[%(asctime)s] [%(levelname)8s] %(name)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 서드파티 라이브러리 로그 레벨 조정
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def setup_cors(app):
    """CORS를 설정합니다."""
    CORS(
        app,
        resources={
            r"/api/*": {
                "origins": os.getenv("CORS_ORIGINS", "*"),
                "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
                "allow_headers": ["Content-Type", "Authorization"],
            }
        },
    )


def setup_swagger(app):
    """Swagger를 설정합니다."""
    swagger_config = {
        "headers": [],
        "specs": [
            {
                "endpoint": "apispec",
                "route": "/apispec.json",
                "rule_filter": lambda rule: True,
                "model_filter": lambda tag: True,
            }
        ],
        "static_url_path": "/flasgger_static",
        "swagger_ui": True,
        "specs_route": "/docs/",
    }
    Swagger(app, config=swagger_config)


# 외부 호환을 위한 간소화된 헬퍼 함수는 제거됨
# build_db_images 함수는 app.utils 모듈에서 직접 import하여 사용
