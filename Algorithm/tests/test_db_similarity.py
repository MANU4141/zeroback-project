import sys, os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recommender.db_similarity import recommend_similar_images


def test_db_similarity():
    # 예시 쿼리 속성
    query = {
        "category": ["셔츠"],
        "color": ["블루"],
        "material": ["면"],
        "detail": ["심플"],
    }
    # 예시 DB
    db_images = [
        {
            "img_path": "static/db/001.jpg",
            "label": {
                "category": ["셔츠"],
                "color": ["블루"],
                "material": ["면"],
                "detail": ["심플"],
            },
        },
        {
            "img_path": "static/db/002.jpg",
            "label": {
                "category": ["블라우스"],
                "color": ["화이트"],
                "material": ["쉬폰"],
                "detail": ["프릴"],
            },
        },
    ]
    sims = recommend_similar_images(query, db_images, top_n=1)
    print("유사 이미지:", sims[0])
    assert sims[0][0] > 0  # 유사도 점수 양수일 것


if __name__ == "__main__":
    test_db_similarity()
    print("DB 유사도 추천 테스트 OK")
