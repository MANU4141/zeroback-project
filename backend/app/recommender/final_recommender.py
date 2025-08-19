import logging
import os
from app.weather import KoreaWeatherAPI
from app.llm.gemini_prompt_utils import analyze_user_prompt
from app.recommender.style_mappings import extract_style_preferences
from collections import Counter
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor, TimeoutError


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


def _score_image(
    label: Dict[str, str],
    targets: Dict[str, List[str]],
    ai_categories: List[str] = None,
    user_style_weight: float = 1.0,
) -> float:
    score = 0.0

    # AI 카테고리 확인 (드레스 등)
    ai_categories = ai_categories or []
    label_categories = label.get("category", [])
    if isinstance(label_categories, str):
        label_categories = [label_categories]

    for key in (
        "category",
        "color",
        "sleeve_length",
        "neckline",
        "fit",
        "style",
        "material",
        "print",
        "detail",
        "collar",
    ):
        vals = label.get(key, "")
        if not vals:
            continue

        # 리스트와 문자열 모두 처리 가능하도록 수정
        if isinstance(vals, list):
            label_set = set(str(v).strip() for v in vals if v)  # 리스트인 경우
        elif isinstance(vals, str):
            label_set = set(map(str.strip, vals.split(",")))  # 문자열인 경우
            label_set.discard("")  # 빈 문자열 제거
        else:
            label_set = set()

        target_list = targets.get(key, []) or []
        target_set = set(target_list)
        if not target_set:
            continue
        inter = label_set & target_set
        if inter:
            # weight per field - 카테고리 우선, 사용자 요청 반영
            base_weight = {
                "category": 15,  # 카테고리 매칭 절대 우선
                "style": 4,  # 스타일 매칭 강화 (사용자 요청 반영)
                "color": 3,  # 색상 매칭 강화
                "detail": 2,  # 디테일 매칭 강화
                "print": 2,  # 프린트 매칭 강화
                "material": 2,  # 소재 매칭
                "fit": 1.5,  # 핏 매칭
                "neckline": 1.5,  # 넥라인 매칭
                "sleeve_length": 1.5,  # 소매길이 매칭
                "collar": 1,  # 칼라 매칭
            }.get(key, 1)

            # 사용자 스타일 선호도 관련 속성에 추가 가중치 부여
            if key in ["style", "color", "detail", "print"] and user_style_weight > 1.0:
                base_weight *= 1.5  # 적당한 추가 가중치            # AI 카테고리와 다른 카테고리 매칭 시 가중치 감소
            if key == "category":
                # AI 카테고리(드레스)와 완전히 다른 카테고리(셔츠)면 점수 대폭 감소
                if ai_categories and not any(cat in ai_categories for cat in label_set):
                    base_weight = 0.1  # 카테고리 불일치 시 페널티

            score += base_weight * len(inter)

            # 카테고리 완전 일치 시 강력한 추가 보너스 (드레스→드레스 등)
            if key == "category" and len(inter) > 0 and base_weight > 1:
                score += 5  # 카테고리 매칭 강력 보너스 (페널티 받은 경우 제외)
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

    # 입력 값 정규화 (NPE 방지)
    ai_attributes = ai_attributes or {}
    style_preferences = style_preferences or []
    db_images = db_images or []

    try:
        weather_attrs = _local_weather_rules(weather)
        prompt_info = analyze_user_prompt(
            user_prompt, style_preferences, api_key=gemini_api_key
        )

        # 사용자 스타일 선호도 추출 (개선된 부분)
        user_style_prefs = extract_style_preferences(user_prompt, prompt_info)

        # 디버그 정보 제거 (프로덕션에서는 불필요)
        user_style_prefs.pop("_debug_scores", None)

        style_weight = user_style_prefs.get("weight_multiplier", 1.0)

        logger.info(f"사용자 스타일 선호도: {user_style_prefs}")
        logger.info(f"스타일 가중치: {style_weight}")

        def weighted_union(*weighted_lists):
            c = Counter()
            for attr_list, weight in weighted_lists:
                for attr in set(attr_list or []):
                    c[attr] += weight
            return [attr for attr, _ in c.most_common()]

        # 가중치 조정 - AI 분석한 카테고리를 최우선으로 하되, 사용자 요청도 반영
        categories = weighted_union(
            (
                ai_attributes.get("category", []) if ai_attributes else [],
                10,
            ),  # AI 카테고리 최우선
            (user_style_prefs.get("style", []), 2),  # 사용자 스타일은 보완적 역할
            (prompt_info.get("keywords", []), 1),
            (weather_attrs.get("category", []), 0.5),  # 날씨는 최소 영향
        )
        materials = weighted_union(
            (ai_attributes.get("material", []) if ai_attributes else [], 2),
            (user_style_prefs.get("material", []), 2 * style_weight),
            (weather_attrs.get("material", []), 1),
        )
        colors = weighted_union(
            (ai_attributes.get("color", []) if ai_attributes else [], 2),
            (user_style_prefs.get("color", []), 3 * style_weight),  # 색상 선호도 강화
            (prompt_info.get("color_preferences", []), 2),
            (weather_attrs.get("color", []), 1),
        )
        details = weighted_union(
            (ai_attributes.get("detail", []) if ai_attributes else [], 2),
            (user_style_prefs.get("detail", []), 2.5 * style_weight),  # 디테일 강화
            (weather_attrs.get("detail", []), 1),
        )

        # LLM 호출 with 5초 타임아웃
        def call_llm_with_timeout():
            try:
                import google.generativeai as genai

                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel("gemini-1.5-flash")
                llm_prompt = f"""
당신은 패션 스타일리스트입니다. 다음 정보를 바탕으로 개인화된 의상 추천을 해주세요.

**사용자 요청**: "{user_prompt}"
**선택한 스타일**: {', '.join(style_preferences) if style_preferences else '없음'}
**오늘 날씨**: {weather.get('temperature', '')}°C, {weather.get('condition', '')}

**분석된 사용자 선호도**:
- 스타일 키워드: {', '.join(user_style_prefs.get('style', [])[:3])}
- 선호 색상: {', '.join(user_style_prefs.get('color', [])[:3])}
- 선호 디테일: {', '.join(user_style_prefs.get('detail', [])[:3])}

**AI가 분석한 이미지**: {', '.join(ai_attributes.get('style', [])[:2]) if ai_attributes else '없음'}

**추천 근거**:
1. 날씨 적합성: {', '.join(categories[:2])}
2. 스타일 매칭: {', '.join(user_style_prefs.get('style', [])[:2])}
3. 색상 조합: {', '.join(colors[:2])}

한 문장으로 따뜻하고 개인화된 추천 메시지를 작성해주세요. 사용자의 요청("{user_prompt}")을 반드시 언급하세요.
"""
                response = model.generate_content(llm_prompt)
                return response.text.strip()
            except Exception as e:
                raise e

        try:
            # 5초 타임아웃으로 LLM 호출
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(call_llm_with_timeout)
                recommendation_text = future.result(timeout=5.0)
                logger.info("LLM 추천 텍스트 생성 성공")
        except TimeoutError:
            logger.warning(
                "LLM 호출이 5초 타임아웃에 걸렸습니다. 폴백 메시지를 사용합니다."
            )
            cats = categories[:2] or weather_attrs.get("category", [])[:2]
            recommendation_text = (
                f'"{user_prompt}"를 위해 오늘 {weather.get("temperature","")}°C {weather.get("condition","")} 날씨에 '
                f'{", ".join(user_style_prefs.get("style", ["적절한"])[:2])} 스타일의 '
                f'{", ".join(cats)}을(를) 추천합니다.'
            )
        except Exception as e:
            logger.warning(f"LLM 추천 메시지 실패: {e}")
            cats = categories[:2] or weather_attrs.get("category", [])[:2]
            recommendation_text = (
                f"오늘 {weather.get('temperature','')}°C {weather.get('condition','')} 날씨에는 "
                f"{', '.join(cats)}을(를) 추천합니다."
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
            # 모든 AI 속성 추가
            "sleeve_length": (
                ai_attributes.get("sleeve_length", []) if ai_attributes else []
            ),
            "neckline": ai_attributes.get("neckline", []) if ai_attributes else [],
            "fit": ai_attributes.get("fit", []) if ai_attributes else [],
            "print": ai_attributes.get("print", []) if ai_attributes else [],
            "collar": ai_attributes.get("collar", []) if ai_attributes else [],
        }

        # AI 카테고리만 별도로 추출
        ai_categories = ai_attributes.get("category", []) if ai_attributes else []

        scored = []
        for entry in db_images:
            label = entry.get("label", {})
            s = _score_image(
                label, target_pool, ai_categories, style_weight
            )  # 사용자 스타일 가중치 추가
            if s > 0:
                scored.append(
                    {
                        "img_path": entry.get("img_path"),
                        "label": label,
                        "similarity_score": s,
                    }
                )
        scored.sort(key=lambda x: x["similarity_score"], reverse=True)

        # 상위 100개 이미지 이름과 스코어만 추출
        top_100_images = []
        for item in scored[:100]:
            img_path = item.get("img_path", "")
            img_name = os.path.basename(img_path or "")
            top_100_images.append(
                {"image_name": img_name, "score": item.get("similarity_score", 0)}
            )

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
            "image_list": top_100_images,
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
            "image_list": [],
            "total_pages": 0,
            "page": 1,
        }
