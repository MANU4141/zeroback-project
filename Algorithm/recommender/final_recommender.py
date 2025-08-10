import math
from collections import Counter
import logging

# backend. 모듈 경로 제거 (상대 임포트 사용)
from weather.recommend_by_weather import recommend_by_weather
from llm.gemini_prompt_utils import analyze_user_prompt

# 로거 설정
logger = logging.getLogger(__name__)


def _calculate_weighted_attributes(ai_attributes, prompt_info, weather_attrs):
    """가중치를 적용하여 속성들의 우선순위를 계산합니다."""

    def weighted_union(*weighted_lists):
        """가중치가 적용된 리스트들을 통합하여 가장 흔한 아이템 순으로 정렬된 리스트를 반환합니다."""
        counter = Counter()
        for attr_list, weight in weighted_lists:
            if attr_list:  # 리스트가 비어있지 않은 경우에만 업데이트
                counter.update({attr: weight for attr in attr_list})
        return [attr for attr, _ in counter.most_common()]

    # AI 분석 결과가 있을 때와 없을 때를 모두 처리
    ai_cat = ai_attributes.get("category", []) if ai_attributes else []
    ai_mat = ai_attributes.get("material", []) if ai_attributes else []
    ai_col = ai_attributes.get("color", []) if ai_attributes else []
    ai_det = ai_attributes.get("detail", []) if ai_attributes else []

    # 각 속성별 가중치 적용하여 통합
    # 카테고리: AI 분석(3) > 사용자 프롬프트(2) > 날씨(1)
    categories = weighted_union(
        (ai_cat, 3),
        (prompt_info.get("keywords", []), 2),
        (weather_attrs.get("category", []), 1),
    )
    # 소재: AI 분석(2) > 날씨(1)
    materials = weighted_union(
        (ai_mat, 2),
        (weather_attrs.get("material", []), 1),
    )
    # 색상: AI 분석(2) > 사용자 프롬프트(2) > 날씨(1)
    colors = weighted_union(
        (ai_col, 2),
        (prompt_info.get("color_preferences", []), 2),
        (weather_attrs.get("color", []), 1),
    )
    # 디테일: AI 분석(2) > 날씨(1)
    details = weighted_union(
        (ai_det, 2),
        (weather_attrs.get("detail", []), 1),
    )

    return {
        "categories": categories,
        "materials": materials,
        "colors": colors,
        "details": details,
    }


def _calculate_similarity_scores(db_images, style_preferences, user_attrs):
    """DB의 모든 이미지에 대해 유사도 점수를 계산하고 정렬합니다."""
    all_scored_images = []
    style_set = set(style_preferences or [])

    # 비교에 사용할 속성 키
    compare_keys = ["style", "category", "color", "material", "detail"]

    for img_data in db_images:
        label = img_data.get("label", {})
        total_score = 0

        # 1. 스타일 매칭 (가장 높은 가중치)
        img_styles_str = label.get("style", "")
        if img_styles_str:
            img_styles = set(s.strip() for s in img_styles_str.split(","))
            if style_set and style_set.intersection(img_styles):
                total_score += 10

        # 2. 다른 속성들 매칭
        for key in compare_keys:
            if key == "style":
                continue  # 스타일은 위에서 처리했으므로 건너뜀

            user_val_set = user_attrs.get(key, set())
            img_val_str = label.get(key, "")

            if user_val_set and img_val_str:
                img_val_set = set(v.strip() for v in img_val_str.split(","))
                intersection_count = len(user_val_set.intersection(img_val_set))

                if intersection_count > 0:
                    # 가중치: 카테고리/색상(3) > 소재/디테일(2)
                    weight = 3 if key in ["category", "color"] else 2
                    total_score += intersection_count * weight

        if total_score > 0:
            all_scored_images.append({"score": total_score, "image_data": img_data})

    # 점수가 높은 순으로 정렬
    all_scored_images.sort(key=lambda x: x["score"], reverse=True)
    return all_scored_images


def _generate_recommendation_text(
    weather, user_prompt, top_categories, top_materials, top_colors, gemini_api_key
):
    """LLM을 사용하여 추천 텍스트를 생성합니다."""
    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        llm_prompt = f"""
        당신은 패션 전문가입니다. 아래 정보를 바탕으로 사용자에게 자연스럽고 친절한 스타일 추천 문장을 만들어주세요.

        - 오늘의 날씨: 기온 {weather.get('temperature', 'N/A')}°C, 상태 {weather.get('condition', 'N/A')}
        - 사용자 요청: "{user_prompt}"
        - 추천 아이템 키워드:
          - 카테고리: {', '.join(top_categories)}
          - 소재: {', '.join(top_materials)}
          - 색상: {', '.join(top_colors)}

        위 정보를 조합하여, 날씨와 사용자 요청에 맞는 OOTD(Outfit of the Day)를 제안해주세요.
        """
        response = model.generate_content(llm_prompt)
        return response.text.strip()
    except Exception as e:
        logger.warning(f"LLM 추천 메시지 생성 실패: {e}")
        # LLM 실패 시 폴백 메시지 생성
        return (
            f"오늘 같이 {weather.get('temperature','알 수 없는')}°C의 {weather.get('condition','')} 날씨에는 "
            f"{', '.join(top_categories)} 종류의 옷을 추천해 드려요. "
            f"{', '.join(top_colors)} 색상이나 {', '.join(top_materials)} 소재는 어떠신가요?"
        )


def final_recommendation(
    weather,
    user_prompt,
    style_preferences,
    ai_attributes=None,
    gemini_api_key=None,
    db_images=None,
    page=1,
    per_page=3,
):
    """
    모든 입력(날씨, 사용자 요청, AI 분석 결과)을 종합하여 최종 의상 추천을 생성합니다.
    """
    if db_images is None:
        db_images = []

    # 1. 각 소스로부터 속성 추천 받기
    weather_attrs = recommend_by_weather(weather)
    prompt_info = analyze_user_prompt(
        user_prompt, style_preferences, api_key=gemini_api_key
    )

    # 2. 가중치를 적용하여 속성 통합
    weighted_attrs = _calculate_weighted_attributes(
        ai_attributes, prompt_info, weather_attrs
    )

    # 3. 사용자 속성 집합 생성 (유사도 계산용)
    user_attrs_for_similarity = {
        "style": set(style_preferences or []),
        "category": set(weighted_attrs["categories"]),
        "color": set(weighted_attrs["colors"]),
        "material": set(weighted_attrs["materials"]),
        "detail": set(weighted_attrs["details"]),
    }
    if ai_attributes:  # AI 분석 결과가 있으면 사용자 속성에 추가 (가중치 역할)
        for key in ["category", "color", "material", "detail"]:
            if key in ai_attributes:
                user_attrs_for_similarity[key].update(ai_attributes[key])

    # 4. DB 이미지와 유사도 계산 및 정렬
    scored_images = _calculate_similarity_scores(
        db_images, style_preferences, user_attrs_for_similarity
    )

    # 5. 페이지네이션 적용
    total_images = len(scored_images)
    total_pages = math.ceil(total_images / per_page) if per_page > 0 else 0
    start_idx = (page - 1) * per_page
    end_idx = start_idx + per_page
    paginated_images = scored_images[start_idx:end_idx]

    # 6. LLM으로 추천 텍스트 생성
    recommendation_text = _generate_recommendation_text(
        weather,
        prompt_info.get("cleaned_request", user_prompt),
        weighted_attrs["categories"][:3],
        weighted_attrs["materials"][:2],
        weighted_attrs["colors"][:2],
        gemini_api_key,
    )

    # 7. 최종 결과 구성
    return {
        "recommendation_text": recommendation_text,
        "images": [
            {
                "img_path": item["image_data"]["img_path"],
                "similarity_score": item["score"],
                "label": item["image_data"]["label"],
            }
            for item in paginated_images
        ],
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "total_images": total_images,
        # 디버깅 및 추가 정보
        "debug_info": {
            "user_analysis": prompt_info,
            "ai_attributes": ai_attributes,
            "weather_attributes": weather_attrs,
            "weighted_attributes": {
                k: v[:5] for k, v in weighted_attrs.items()
            },  # 상위 5개만 표시
        },
    }
