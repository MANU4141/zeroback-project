"""
Gemini LLM을 활용한 사용자 프롬프트 정리 및 구조화 모듈
"""

from typing import Dict
import google.generativeai as genai
import os

# 환경변수 또는 직접 입력
GEMINI_API_KEY = "secret"  # 보안상 유출 주의, 예시의 키는 그대로 사용 금지
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY  # 환경변수에도 명시적으로 등록


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
    api_key = api_key or GEMINI_API_KEY
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-1.5-flash")

    prompt = f"""
사용자의 의상 추천 요청을 분석하고 정리해주세요.

**사용자 입력:**
{user_prompt}

**선택한 스타일:**
{', '.join(style_preferences) if style_preferences else ''}

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
    try:
        response = model.generate_content(prompt)
        # JSON 부분만 추출
        import json

        text = response.text
        start = text.find("{")
        end = text.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        else:
            return {"cleaned_request": user_prompt}
    except Exception as e:
        print(f"Gemini 분석 오류: {e}")
        return {"cleaned_request": user_prompt}


# 예시 사용법
if __name__ == "__main__":
    prompt = "오늘 데이트라서 귀엽고 깔끔하게 입고 싶어요. 파스텔톤 좋아요."
    result = analyze_user_prompt(prompt, style_preferences=["로맨틱", "캐주얼"])
    print("Gemini 분석 결과:")
    for k, v in result.items():
        print(f"{k}: {v}")
