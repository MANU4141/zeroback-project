import json
import os


def load_db_images():
    """
    db_images.json 파일에서 이미지/라벨 정보를 로드합니다.
    파일이 없으면 빈 리스트 반환.
    """
    db_path = os.path.join(os.path.dirname(__file__), "db_images.json")
    if not os.path.exists(db_path):
        print(f"[DB] db_images.json 파일이 없습니다: {db_path}")
        return []
    with open(db_path, encoding="utf-8") as f:
        return json.load(f)
