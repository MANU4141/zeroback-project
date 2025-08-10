try:
    from backend.weather.recommend_by_weather import recommend_by_weather
    from backend.llm.gemini_prompt_utils import analyze_user_prompt
except ImportError:
    from weather.recommend_by_weather import recommend_by_weather
    from llm.gemini_prompt_utils import analyze_user_prompt

from collections import Counter


def final_recommendation(
    weather,
    user_prompt,
    style_preferences,
    ai_attributes=None,
    gemini_api_key=None,
    db_images=None,
):
    weather_attrs = recommend_by_weather(weather)
    prompt_info = analyze_user_prompt(
        user_prompt, style_preferences, api_key=gemini_api_key
    )

    def weighted_union(*weighted_lists):
        c = Counter()
        for attr_list, weight in weighted_lists:
            c.update({attr: weight for attr in attr_list})
        return [attr for attr, _ in c.most_common()]

    categories = weighted_union(
        (ai_attributes.get("category", []) if ai_attributes else [], 3),
        (prompt_info.get("keywords", []), 2),
        (weather_attrs.get("category", []), 1),
    )
    materials = weighted_union(
        (ai_attributes.get("material", []) if ai_attributes else [], 2),
        (weather_attrs.get("material", []), 1),
    )
    colors = weighted_union(
        (ai_attributes.get("color", []) if ai_attributes else [], 2),
        (prompt_info.get("color_preferences", []), 2),
        (weather_attrs.get("color", []), 1),
    )
    details = weighted_union(
        (ai_attributes.get("detail", []) if ai_attributes else [], 2),
        (weather_attrs.get("detail", []), 1),
    )

    # 스타일 기반 + 속성 유사도 기반 top-N 추천
    recommended_images = []
    style_matched = []
    similarity_ranked = []
    db_images = db_images or []
    # 비교 기준 속성
    compare_keys = ["style", "category", "color", "material", "detail"]
    # AI 분석 결과 또는 프롬프트 기반 속성 추출
    user_attrs = {}
    if ai_attributes:
        for k in compare_keys:
            v = ai_attributes.get(k)
            if v:
                user_attrs[k] = set(v if isinstance(v, list) else [v])
    # style 우선 매칭
    style_set = set(style_preferences or [])
    for img in db_images:
        label = img.get("label", {})
        img_styles = set(label.get("style", []))
        if style_set and (style_set & img_styles):
            style_matched.append(img)
        else:
            # 속성 유사도 계산 (style 제외, 나머지 속성 일치 개수)
            sim = 0
            for k in compare_keys:
                if k == "style":
                    continue
                user_v = user_attrs.get(k, set())
                img_v = set(label.get(k, []))
                sim += len(user_v & img_v)
            similarity_ranked.append((sim, img))
    # style 일치 이미지 우선, 그 외에는 유사도 순 정렬
    similarity_ranked.sort(key=lambda x: x[0], reverse=True)
    # top-N 추천 (style 일치 + 유사도 높은 순)
    N = 3
    recommended_images = style_matched[:N]
    if len(recommended_images) < N:
        recommended_images += [
            img for _, img in similarity_ranked[: N - len(recommended_images)]
        ]
    # style_matched_images는 style 일치만, all_recommended_images는 top-N
    # (style_matched가 3개 미만이면 유사도 높은 이미지로 채움)

    # 기존 추천 속성(categories 등)과 style 기반 추천을 조합해 최종 추천 리스트 생성
    # 실제 응답에 recommended_images를 활용하려면 아래 반환 dict에 추가
    try:
        import google.generativeai as genai

        genai.configure(api_key=gemini_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        llm_prompt = f"""
오늘의 날씨: {weather.get('temperature', '')}°C, {weather.get('condition', '')}
사용자 요청: {prompt_info.get('cleaned_request', user_prompt)}
추천 속성: {', '.join(categories[:3])}, {', '.join(materials[:2])}, {', '.join(colors[:2])}
"""
        response = model.generate_content(llm_prompt)
        recommendation_text = response.text.strip()
    except Exception as e:
        print(f"[Warning] LLM 추천 메시지 실패: {e}")
        recommendation_text = f"오늘 {weather.get('temperature','')}°C {weather.get('condition','')} 날씨에는 {', '.join(categories[:3])}를 추천합니다."
    return {
        "recommendation_text": recommendation_text,
        "suggested_items": categories[:3],
        "weather": weather,
        "user_analysis": prompt_info,
        "ai_attributes": ai_attributes,
        "weather_attributes": weather_attrs,
        "materials": materials[:2],
        "colors": colors[:2],
        "details": details[:2],
        "style_matched_images": style_matched,  # style 매칭된 이미지 리스트
        "all_recommended_images": recommended_images,  # top-N 추천 이미지 리스트
    }
