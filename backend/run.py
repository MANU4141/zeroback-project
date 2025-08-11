# backend/run.py
import os
import sys

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app import create_app
from config import Config

# 애플리케이션 인스턴스 생성
app = create_app()

if __name__ == "__main__":
    # 환경변수에서 포트와 호스트를 가져오거나 기본값 사용
    port = int(os.environ.get("FLASK_RUN_PORT", Config.FLASK_RUN_PORT))
    host = os.environ.get("FLASK_RUN_HOST", Config.FLASK_RUN_HOST)
    is_debug = os.environ.get("FLASK_DEBUG", str(Config.FLASK_DEBUG)).lower() == "true"

    # Flask 앱 실행
    app.run(host=host, port=port, debug=is_debug)
