try:
    from backend.weather.recommend_by_weather import recommend_by_weather
    from backend.llm.gemini_prompt_utils import analyze_user_prompt
except ImportError:
    from weather.recommend_by_weather import recommend_by_weather
    from llm.gemini_prompt_utils import analyze_user_prompt

from collections import Counter

def final_recommendation(
    weather, user_prompt, style_preferences, ai_attributes=None, gemini_api_key=None
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
    }
