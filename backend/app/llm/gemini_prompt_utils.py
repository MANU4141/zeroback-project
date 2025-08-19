"""
Gemini LLM을 활용한 사용자 프롬프트 정리 및 구조화 모듈
"""

import logging
from typing import Dict
import google.generativeai as genai
import os

logger = logging.getLogger(__name__)

# 환경변수에서 API 키 읽기
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
    genai.configure(api_key=GEMINI_API_KEY)


def analyze_user_prompt(
    user_prompt: str, style_preferences: list = None, api_key: str = None
) -> Dict:
    """
    Gemini LLM을 활용해 사용자 프롬프트를 정리/구조화
    Args:
        user_prompt: 사용자 자유 입력 텍스트
        style_preferences: 사용자가 선택한 스타일 리스트(선택)
        api_key: Gemini API 키(선택)
    Returns:
        {
            "keywords": [...],
            "mood": str,
            "occasion": str,
            "special_requests": [...],
            "color_preferences": [...],
            "cleaned_request": str
        }
    """
    # 기본 프롬프트 가드
    if not user_prompt or not user_prompt.strip():
        user_prompt = "편안하고 일상적인 의상을 추천해주세요"

    # style_preferences None/빈 리스트 처리 일관화
    if not style_preferences:
        style_preferences = []

    # genai.configure 중복/키 누락 방지
    effective_api_key = api_key or GEMINI_API_KEY
    if not effective_api_key:
        logger.warning("Gemini API 키가 설정되지 않았습니다")
        return _get_fallback_response(user_prompt)

    try:
        genai.configure(api_key=effective_api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        prompt = f"""
사용자의 의상 추천 요청을 분석하고 정리해주세요.

**사용자 입력:**
{user_prompt.strip()}

**선택한 스타일:**
{', '.join(style_preferences) if style_preferences else '없음'}

다음 정보를 추출하여 JSON 형식으로 응답해주세요:
1. 주요 키워드
2. 무드
3. 상황
4. 특별 요청
5. 색상 선호도
6. 정리된 요청(한 문장)

JSON 예시:
{{
"keywords": ["키워드1", "키워드2"],
"mood": "무드 설명",
"occasion": "상황 설명",
"special_requests": ["특별 요청1"],
"color_preferences": ["색상1"],
"cleaned_request": "정리된 요청 문장"
}}
"""

        response = model.generate_content(prompt)
        return _parse_gemini_response(response.text, user_prompt)

    except Exception as e:
        logger.exception("Gemini 분석 중 오류 발생")
        return _get_fallback_response(user_prompt)


def _parse_gemini_response(response_text: str, fallback_prompt: str) -> Dict:
    """출력 파싱 견고화: JSON 코드블록, markdown fence, 텍스트 섞임 처리"""
    import json

    # JSON 블록 추출 시도 (여러 패턴)
    patterns_to_try = [
        # 일반적인 { ... } 블록
        (response_text.find("{"), response_text.rfind("}") + 1),
        # 코드블록 내부 찾기
        (
            response_text.find("```json"),
            response_text.find("```", response_text.find("```json") + 7),
        ),
        (
            response_text.find("```"),
            response_text.find("```", response_text.find("```") + 3),
        ),
    ]

    for start_marker, end_marker in patterns_to_try:
        if start_marker != -1 and end_marker > start_marker:
            try:
                # JSON 블록만 추출
                if "```json" in response_text:
                    json_text = response_text[start_marker + 7 : end_marker].strip()
                elif "```" in response_text:
                    json_text = response_text[start_marker + 3 : end_marker].strip()
                else:
                    json_text = response_text[start_marker:end_marker].strip()

                # { ... } 블록 재추출
                json_start = json_text.find("{")
                json_end = json_text.rfind("}") + 1
                if json_start != -1 and json_end > json_start:
                    json_text = json_text[json_start:json_end]

                parsed_result = json.loads(json_text)
                return _fill_missing_fields(parsed_result, fallback_prompt)
            except json.JSONDecodeError:
                continue

    # 모든 파싱 시도 실패 시 폴백
    logger.warning("Gemini 응답 JSON 파싱 실패, 폴백 응답 사용")
    return _get_fallback_response(fallback_prompt)


def _fill_missing_fields(parsed_result: dict, fallback_prompt: str) -> Dict:
    """반환 스키마 필드 누락 시 기본값 채우기"""
    default_schema = {
        "keywords": [],
        "mood": "일상적",
        "occasion": "데일리",
        "special_requests": [],
        "color_preferences": [],
        "cleaned_request": fallback_prompt,
    }

    # 누락된 필드를 기본값으로 채움
    for key, default_value in default_schema.items():
        if key not in parsed_result:
            parsed_result[key] = default_value

    return parsed_result


def _get_fallback_response(user_prompt: str) -> Dict:
    """API 실패 시 기본 응답"""
    return {
        "keywords": ["기본", "데일리"],
        "mood": "일상적",
        "occasion": "데일리",
        "special_requests": [],
        "color_preferences": [],
        "cleaned_request": user_prompt.strip() or "편안한 의상을 추천해주세요",
    }


# 예시 사용법
if __name__ == "__main__":
    prompt = "오늘 데이트라서 귀엽고 깔끔하게 입고 싶어요. 파스텔톤 좋아요."
    result = analyze_user_prompt(prompt, style_preferences=["로맨틱", "캐주얼"])
    print("Gemini 분석 결과:")
    for k, v in result.items():
        print(f"{k}: {v}")
