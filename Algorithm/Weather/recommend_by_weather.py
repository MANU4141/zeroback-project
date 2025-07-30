# Weather/recommend_by_weather.py

from typing import Dict, List


def recommend_by_weather(weather: Dict) -> Dict[str, List[str]]:
    """
    날씨 정보(온도, 상태, 습도, 풍속 등)를 받아
    실제 서비스 추천 파이프라인에서 사용할 수 있는
    카테고리/색상/소재/디테일 등 추천 속성 dict를 반환합니다.

    Args:
        weather: {
            "temperature": float,      # 온도(°C)
            "condition": str,          # 예: "맑음", "흐림", "비", "눈"
            "humidity": int,           # 습도(%)
            "wind_speed": float        # 풍속(m/s)
        }

    Returns:
        추천 속성 dict 예:
        {
            "category": [...],
            "material": [...],
            "color": [...],
            "detail": [...]
        }
    """
    temp = weather.get("temperature")
    condition = weather.get("condition", "")
    humidity = weather.get("humidity", 50)
    wind_speed = weather.get("wind_speed", 2)

    categories, materials, colors, details = [], [], [], []

    # 온도별 추천
    if temp is not None:
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

    # 날씨 상태별 추천(디테일, 소재, 색상)
    if "비" in condition or "눈" in condition:
        details += ["방수", "우산", "레인부츠"]
        materials += ["비닐/PVC", "방수 소재"]
    if "맑음" in condition:
        details += ["선글라스"]
        colors += ["화이트", "옐로우", "스카이블루"]
    if "흐림" in condition:
        details += ["레이어드"]
        materials += ["면", "니트웨어"]

    # 습도/풍속 보정
    if humidity >= 80:
        materials += ["린넨", "메시"]
        details += ["통기성"]
    if wind_speed >= 6:
        categories += ["바람막이", "점퍼"]
        details += ["모자"]

    # 중복 제거 및 정렬
    return {
        "category": sorted(set(categories)),
        "material": sorted(set(materials)),
        "color": sorted(set(colors)),
        "detail": sorted(set(details)),
    }


# =============================================================================
# 예시 입력/출력 (실제 서비스 운영에서 사용되는 포맷, 아래처럼 참고)
# =============================================================================
"""
입력 예시:
{
    "temperature": 25.0,
    "condition": "맑음",
    "humidity": 60,
    "wind_speed": 3
}

출력 예시:
{
    "category": ["가벼운 아우터", "반팔", "셔츠"],
    "material": ["린넨", "면"],
    "color": ["베이지", "블루", "스카이블루", "화이트", "옐로우"],
    "detail": ["선글라스"]
}
"""
