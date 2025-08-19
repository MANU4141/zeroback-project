# backend/run.py
import os
import sys
import logging

# 프로젝트 루트 디렉토리를 Python 경로에 추가 (중복 방지)
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from app import create_app
from config import Config, _parse_int_env, _parse_bool_env

logger = logging.getLogger(__name__)

# --- 상수 정의 ---
VALID_PORT_MIN = 1024
VALID_PORT_MAX = 65535
DEFAULT_PORT = 5000


def get_server_config():
    """서버 설정을 안전하게 가져옵니다."""
    try:
        # Config 헬퍼 함수 사용으로 파싱 일관화
        port = _parse_int_env("FLASK_RUN_PORT", Config.FLASK_RUN_PORT)
        host = os.environ.get("FLASK_RUN_HOST", Config.FLASK_RUN_HOST)
        debug = _parse_bool_env("FLASK_DEBUG", Config.FLASK_DEBUG)

        # 포트 범위 검증 (상수 사용)
        if not (VALID_PORT_MIN <= port <= VALID_PORT_MAX):
            logger.warning(
                f"포트 {port}가 유효 범위({VALID_PORT_MIN}-{VALID_PORT_MAX})를 벗어남. 기본값 {DEFAULT_PORT} 사용"
            )
            port = DEFAULT_PORT

        return host, port, debug
    except Exception as e:
        logger.exception("서버 설정 파싱 오류, 기본값 사용")
        return Config.FLASK_RUN_HOST, Config.FLASK_RUN_PORT, Config.FLASK_DEBUG


def main():
    """메인 실행 함수"""
    # 설정 검증
    config_errors = Config.validate_config()
    if config_errors:
        logger.warning("설정에 문제가 있지만 서버를 시작합니다")

    # 애플리케이션 생성
    global app  # 모듈 레벨에서 접근 가능하도록
    app = create_app()

    # 서버 설정 가져오기
    host, port, debug = get_server_config()

    logger.info(f"Flask 서버 시작: {host}:{port} (디버그 모드: {debug})")

    # Flask 앱 실행
    try:
        app.run(host=host, port=port, debug=debug)
    except KeyboardInterrupt:
        logger.info("서버가 사용자에 의해 중단되었습니다")
    except Exception as e:
        logger.exception("서버 실행 중 오류 발생")
        sys.exit(1)


# 앱 인스턴스를 모듈 레벨에서 생성 (테스트/import 용도)
app = create_app()


if __name__ == "__main__":
    main()
