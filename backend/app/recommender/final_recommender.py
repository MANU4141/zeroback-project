import logging
from app.weather import KoreaWeatherAPI
from app.llm.gemini_prompt_utils import analyze_user_prompt
from collections import Counter
from typing import List, Dict, Any, Optional


def _local_weather_rules(weather: dict) -> dict:
    """Lightweight weather-to-attributes mapping used inside backend only."""
    temp = weather.get("temperature", 20)
    cond = weather.get("condition", "")
    categories, materials, colors, details = [], [], [], []
    if temp >= 28:
        categories += ["티셔츠", "민소매", "브라탑"]
        materials += ["린넨", "면", "메시"]
        colors += ["화이트", "스카이블루", "옐로우", "민트"]
    elif temp >= 23:
        categories += ["반팔", "셔츠", "가벼운 아우터"]
        materials += ["면", "린넨"]
        colors += ["화이트", "블루", "베이지"]
    elif temp >= 17:
        categories += ["긴팔", "가디건", "니트웨어"]
        materials += ["면", "니트"]
        colors += ["네이비", "브라운", "카키"]
    elif temp >= 12:
        categories += ["니트웨어", "맨투맨", "점퍼", "바람막이"]
        materials += ["니트", "우븐"]
        colors += ["그레이", "블랙", "베이지"]
    elif temp >= 5:
        categories += ["코트", "패딩", "머플러"]
        materials += ["울/캐시미어", "패딩", "플리스"]
        colors += ["블랙", "브라운", "와인"]
    else:
        categories += ["패딩", "목도리", "장갑", "내복"]
        materials += ["패딩", "울/캐시미어", "플리스"]
        colors += ["블랙", "그레이", "네이비"]

    if "비" in cond or "눈" in cond:
        details += ["방수", "우산", "레인부츠"]
        materials += ["비닐/PVC", "방수 소재"]
    if "맑음" in cond:
        details += ["선글라스"]
        colors += ["화이트", "옐로우", "스카이블루"]
    if "흐림" in cond:
        details += ["레이어드"]
        materials += ["면", "니트웨어"]

    return {
        "category": sorted(set(categories)),
        "material": sorted(set(materials)),
        "color": sorted(set(colors)),
        "detail": sorted(set(details)),
    }


def _score_image(label: Dict[str, str], targets: Dict[str, List[str]]) -> float:
    score = 0.0
    for key in ("category", "material", "color", "detail", "style"):
        vals = label.get(key, "")
        if not vals:
            continue
        # label values are comma-separated string from DB build
        label_set = (
            set(map(str.strip, vals.split(","))) if isinstance(vals, str) else set()
        )
        target_list = targets.get(key, []) or []
        target_set = set(target_list)
        if not target_set:
            continue
        inter = label_set & target_set
        if inter:
            # weight per field
            weight = {
                "category": 3,
                "style": 2,
                "material": 1.5,
                "color": 1,
                "detail": 0.5,
            }.get(key, 1)
            score += weight * len(inter)
    return score


def final_recommendation(
    weather: Dict[str, Any],
    user_prompt: str,
    style_preferences: List[str],
    ai_attributes: Optional[Dict[str, List[str]]] = None,
    gemini_api_key: Optional[str] = None,
    db_images: Optional[List[Dict[str, Any]]] = None,
    page: int = 1,
    per_page: int = 3,
):
    logger = logging.getLogger(__name__)
    try:
        weather_attrs = _local_weather_rules(weather)
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
            
            # LLM 프롬프트를 더 명확하고 구체적으로 수정
            llm_prompt = f"""
아래 정보를 바탕으로 스타일링 팁을 추천해주세요.
[정보]
- 날씨: {weather.get('temperature', '')}°C, {weather.get('condition', '')}
- 사용자 요청: {prompt_info.get('cleaned_request', user_prompt)}
- 추천 키워드: {', '.join(categories[:3])}, {', '.join(materials[:2])}, {', '.join(colors[:2])}
[규칙]
- 반드시 한글로, 자연스러운 문장으로 작성해주세요.
- 전체 분량은 350자에서 400자 사이로 맞춰주세요. 이 글자 수 제한을 반드시 지켜야 합니다.
- 응답에 '**'와 같은 마크다운 특수기호는 절대 사용하지 마세요.
- 마지막 문장은 반드시 마침표(.)로 끝나도록 자연스럽게 마무리해주세요.
"""
            response = model.generate_content(llm_prompt)
            recommendation_text = response.text.strip()

            # 후처리 로직 강화 (안전장치)
            # 1. 특수기호 제거
            recommendation_text = recommendation_text.replace("**", "")

            # 2. 길이 초과 시 문장 단위로 자르기
            if len(recommendation_text) > 450:
                # 450자 이내에서 마지막 마침표를 찾음
                last_period_index = recommendation_text.rfind('.', 0, 450)
                if last_period_index != -1:
                    # 마침표를 포함하여 문장을 자름
                    recommendation_text = recommendation_text[:last_period_index + 1]
                else:
                    # 마침표가 없다면, 450자에서 그냥 자르고 '...' 추가
                    recommendation_text = recommendation_text[:450] + "..."

        except Exception as e:
            logger.warning(f"LLM 추천 메시지 실패: {e}")
            recommendation_text = (
                f"오늘 {weather.get('temperature','')}°C {weather.get('condition','')} 날씨에는 "
                f"{', '.join(categories[:3])}를 추천합니다."
            )
        # Simple ranking of DB images
        db_images = db_images or []
        target_pool = {
            "category": categories,
            "material": materials,
            "color": colors,
            "detail": details,
            "style": style_preferences
            or (ai_attributes.get("style", []) if ai_attributes else []),
        }
        scored = []
        for entry in db_images:
            label = entry.get("label", {})
            s = _score_image(label, target_pool)
            if s > 0:
                scored.append(
                    {
                        "img_path": entry.get("img_path"),
                        "label": label,
                        "similarity_score": s,
                    }
                )
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)

        total = len(scored)
        total_pages = (total + per_page - 1) // per_page if per_page > 0 else 1
        page = max(1, min(page, total_pages or 1))
        start = (page - 1) * per_page
        end = start + per_page
        page_items = scored[start:end]

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
            "images": page_items,
            "total_pages": total_pages,
            "page": page,
        }
    except Exception as e:
        # 절대 500으로 터지지 않도록 안전 가드
        logger.exception(f"final_recommendation 실패: {e}")
        return {
            "recommendation_text": f"오늘 {weather.get('temperature','')}°C {weather.get('condition','')} 날씨에 맞는 기본 아이템을 추천합니다.",
            "suggested_items": [],
            "weather": weather,
            "user_analysis": {"cleaned_request": user_prompt},
            "ai_attributes": ai_attributes,
            "weather_attributes": _local_weather_rules(weather),
            "materials": [],
            "colors": [],
            "details": [],
            "images": [],
            "total_pages": 0,
            "page": 1,
        }
