import os, sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from recommender.final_recommender import final_recommendation


def test_final_recommendation():
    # 샘플 입력
    weather = {"temperature": 22, "condition": "흐림"}
    user_prompt = "심플한 셔츠나 블루톤이 좋아요."
    style_preferences = ["캐주얼"]
    ai_attributes = {
        "category": ["셔츠"],
        "color": ["블루"],
        "material": ["면"],
        "detail": ["심플"],
    }
    # 추천 함수 호출
    rec = final_recommendation(
        weather,
        user_prompt,
        style_preferences,
        ai_attributes=ai_attributes,
        gemini_api_key=None,  # 실제 API키 연결시 입력
    )
    print("추천 멘트:", rec["recommendation_text"])
    assert "추천합니다" in rec["recommendation_text"]
    assert len(rec["suggested_items"]) > 0
    assert "category" in rec["user_analysis"]


if __name__ == "__main__":
    test_final_recommendation()
    print("추천 통합 테스트 OK")
