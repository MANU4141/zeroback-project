# backend/app/recommender/style_mappings.py
"""
스타일 키워드 매핑 및 가중치 시스템
"""
import re
from collections import defaultdict
from typing import Dict, List, Any, Tuple


def contains_token(text: str, keyword: str) -> bool:
    """토큰 경계 기반 정확한 키워드 매칭 (부분 문자열 과매칭 방지)"""
    if re.search(rf"\b{re.escape(keyword)}\b", text, flags=re.IGNORECASE):
        return True
    return (
        re.search(
            rf"(^|[\s\W]){re.escape(keyword)}($|[\s\W])", text, flags=re.IGNORECASE
        )
        is not None
    )


# 감정/스타일 키워드를 구체적인 패션 속성으로 매핑
STYLE_KEYWORD_MAPPING = {
    "귀엽게": {
        "variants": ["귀여", "cute", "큐트", "깜찍", "러블리", "lovely"],
        "style": ["로맨틱", "걸리시", "페미닌", "캐주얼"],
        "color": ["핑크", "화이트", "파스텔", "베이비블루", "라벤더"],
        "print": ["플로럴", "도트", "체크", "스트라이프"],
        "detail": ["프릴", "리본", "레이스", "러플"],
        "collar": ["피터팬칼라", "라운드넥", "보우칼라"],
        "fit": ["노멀", "슬림"],
        "weight_multiplier": 2.5,  # 높은 가중치
    },
    "깔끔하게": {
        "variants": ["깔끔", "단정", "심플", "simple", "미니멀", "minimal", "정갈"],
        "style": ["미니멀", "클래식", "오피스"],
        "color": ["화이트", "네이비", "베이지", "그레이"],
        "print": ["무지", "미니멀"],
        "detail": ["단순", "깔끔"],
        "fit": ["노멀", "슬림"],
        "weight_multiplier": 2.0,
    },
    "섹시하게": {
        "variants": ["섹시", "sexy", "글래머", "glamour", "매혹적", "시크", "chic"],
        "style": ["섹시", "글래머", "파티"],
        "color": ["블랙", "레드", "와인", "네이비"],
        "detail": ["시스루", "백리스", "홀터넥"],
        "fit": ["타이트", "슬림"],
        "weight_multiplier": 2.5,
    },
    "캐주얼하게": {
        "variants": ["캐주얼", "casual", "편안", "자연스럽", "데일리", "일상"],
        "style": ["캐주얼", "스트리트", "스포티"],
        "color": ["베이지", "카키", "데님", "그레이"],
        "material": ["데님", "면", "니트"],
        "weight_multiplier": 1.8,
    },
    "우아하게": {
        "variants": ["우아", "엘레간트", "elegant", "품위", "고급", "클래시"],
        "style": ["엘레간트", "클래식", "소피스티케이트"],
        "color": ["네이비", "블랙", "와인", "베이지"],
        "material": ["실크", "울", "캐시미어"],
        "weight_multiplier": 2.2,
    },
    "편안하게": {
        "variants": ["편안", "comfortable", "릴렉스", "relax", "여유", "루즈"],
        "style": ["캐주얼", "스포티", "편안"],
        "material": ["면", "니트", "저지"],
        "fit": ["루즈", "오버사이즈", "노멀"],
        "weight_multiplier": 1.5,
    },
}

# 계절/상황별 키워드 매핑
OCCASION_MAPPING = {
    "데이트": {
        "style": ["로맨틱", "페미닌", "엘레간트"],
        "color": ["핑크", "화이트", "파스텔"],
        "weight_multiplier": 2.0,
    },
    "직장": {
        "style": ["오피스", "클래식", "미니멀"],
        "color": ["네이비", "블랙", "화이트", "베이지"],
        "weight_multiplier": 2.2,
    },
    "여행": {
        "style": ["캐주얼", "스포티", "편안"],
        "material": ["면", "기능성"],
        "weight_multiplier": 1.8,
    },
    "파티": {
        "style": ["파티", "글래머", "섹시"],
        "color": ["블랙", "레드", "골드"],
        "weight_multiplier": 2.5,
    },
}


def extract_style_preferences(
    user_request: str, gemini_analysis: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    사용자 요청과 Gemini 분석 결과에서 스타일 선호도를 추출
    다중 키워드 충돌 시 점수 누적 후 Top-K 선택으로 노이즈 제거

    Args:
        user_request: 원본 사용자 요청
        gemini_analysis: Gemini LLM 분석 결과

    Returns:
        추출된 스타일 선호도와 가중치 (점수 기반 정렬)
    """
    # 속성별 점수 맵 초기화
    pref_scores: Dict[str, Dict[str, float]] = {
        "style": defaultdict(float),
        "color": defaultdict(float),
        "print": defaultdict(float),
        "detail": defaultdict(float),
        "collar": defaultdict(float),
        "fit": defaultdict(float),
        "material": defaultdict(float),
    }
    max_weight = 1.0

    def add_mapping(mapping: Dict[str, Any], base_weight: float):
        """매핑 정보를 점수로 누적"""
        nonlocal max_weight
        w = float(mapping.get("weight_multiplier", 1.0)) * base_weight
        max_weight = max(max_weight, w)

        for attr, values in mapping.items():
            if attr in pref_scores and isinstance(values, list):
                for v in set(values):  # 중복 제거로 과누적 방지
                    pref_scores[attr][v] += w

    # 1) 사용자 텍스트에서 직접 키워드 매칭 (주 신호)
    text_lower = user_request.lower()
    for main_keyword, mapping in STYLE_KEYWORD_MAPPING.items():
        all_keywords = [main_keyword] + mapping.get("variants", [])
        if any(contains_token(text_lower, kw) for kw in all_keywords):
            add_mapping(mapping, base_weight=1.0)  # 주 신호: 가중치 1.0

    # 2) Gemini 분석 결과 활용 (보조 신호)
    if gemini_analysis:
        keywords: List[str] = gemini_analysis.get("keywords", [])
        occasion: str = gemini_analysis.get("occasion", "")
        mood: str = gemini_analysis.get("mood", "")

        # Gemini 키워드 매핑 (보조 신호로 낮은 가중치)
        for keyword in keywords:
            for style_key, mapping in STYLE_KEYWORD_MAPPING.items():
                all_style_keywords = [style_key] + mapping.get("variants", [])
                if any(contains_token(keyword.lower(), k) for k in all_style_keywords):
                    add_mapping(mapping, base_weight=0.8)  # 보조 신호: 가중치 0.8

        # 상황별 매핑
        for occ_key, mapping in OCCASION_MAPPING.items():
            if contains_token(occasion.lower(), occ_key):
                add_mapping(mapping, base_weight=1.0)

    # 3) 속성별 Top-K 선택으로 노이즈 컷
    def topk(score_dict: Dict[str, float], k: int = 5) -> List[str]:
        """점수 기준 상위 K개 선택"""
        return [
            item
            for item, _ in sorted(score_dict.items(), key=lambda x: (-x[1], x[0]))[:k]
        ]

    # 최종 preferences 구성 (점수 기반 정렬)
    preferences = {attr: topk(scores, k=5) for attr, scores in pref_scores.items()}
    preferences["weight_multiplier"] = min(
        max_weight, 3.0
    )  # 상한 클램프로 과도한 가중치 방지

    # 디버깅용: 점수 정보 추가 (개발 환경에서만)
    preferences["_debug_scores"] = {
        attr: dict(scores) for attr, scores in pref_scores.items() if scores
    }

    return preferences


def get_style_preferences_with_scores(
    user_request: str, gemini_analysis: Dict[str, Any] = None
) -> Tuple[Dict[str, Any], Dict[str, Dict[str, float]]]:
    """
    스타일 선호도와 상세 점수 정보를 함께 반환 (디버깅용)

    Returns:
        (preferences, detailed_scores)
    """
    result = extract_style_preferences(user_request, gemini_analysis)
    detailed_scores = result.pop("_debug_scores", {})
    return result, detailed_scores
