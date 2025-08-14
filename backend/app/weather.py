import requests
import os
import math
import logging
from typing import Optional
from datetime import datetime, timedelta
from dotenv import load_dotenv

# .env 파일에서 환경 변수 로드
load_dotenv()

# 로거 설정
logging.basicConfig(
    level=logging.INFO, format="[%(asctime)s] [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class KoreaWeatherAPI:
    """
    한국 기상청 단기예보 API를 사용하여 날씨 정보를 조회하는 클래스.
    """

    # 기상청 격자 변환용 상수
    RE = 6371.00877  # 지구 반경(km)
    GRID = 5.0  # 격자 간격(km)
    SLAT1 = 30.0  # 투영 위도 1(degree)
    SLAT2 = 60.0  # 투영 위도 2(degree)
    OLON = 126.0  # 기준점 경도(degree)
    OLAT = 38.0  # 기준점 위도(degree)
    XO = 43  # 기준점 X좌표(GRID)
    YO = 136  # 기준점 Y좌표(GRID)

    def __init__(self):
        """API 클라이언트 초기화."""
        self.service_key = os.getenv("WEATHER_API_KEY_DECODE")
        self.base_url = (
            "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst"
        )

        if not self.service_key:
            logger.warning(
                "기상청 API 키가 설정되지 않았습니다. .env 파일을 확인해주세요."
            )
        else:
            logger.info("기상청 날씨 API 클라이언트가 초기화되었습니다.")

    def get_weather_info(self, latitude: float, longitude: float) -> dict:
        """
        주어진 위경도에 대한 현재 날씨 정보를 조회합니다.
        API 호출 실패 시 폴백 데이터를 반환합니다.
        """
        if not self.service_key:
            logger.error("API 키가 없어 날씨 정보를 조회할 수 없습니다.")
            return self.get_fallback_weather()

        try:
            logger.info(f"날씨 정보 조회 시작: lat={latitude}, lon={longitude}")

            grid_x, grid_y = self._convert_coords_to_grid(latitude, longitude)
            logger.info(f"격자 변환 완료: X={grid_x}, Y={grid_y}")

            base_date, base_time = self._get_forecast_time()
            logger.info(f"예보 기준 시간: {base_date} {base_time}")

            weather_data = self._call_weather_api(grid_x, grid_y, base_date, base_time)

            result = self._parse_weather_data(weather_data)
            logger.info(f"최종 날씨 정보: {result}")
            return result

        except Exception as e:
            logger.error(f"날씨 정보 조회 중 오류 발생: {e}", exc_info=True)
            return self.get_fallback_weather()

    def _convert_coords_to_grid(self, lat: float, lon: float) -> tuple[int, int]:
        """위경도를 기상청 격자 좌표(X, Y)로 변환합니다."""
        DEGRAD = math.pi / 180.0

        re = self.RE / self.GRID
        slat1 = self.SLAT1 * DEGRAD
        slat2 = self.SLAT2 * DEGRAD
        olon = self.OLON * DEGRAD
        olat = self.OLAT * DEGRAD

        sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(
            math.tan(math.pi * 0.25 + slat2 * 0.5)
            / math.tan(math.pi * 0.25 + slat1 * 0.5)
        )
        sf = math.pow(math.tan(math.pi * 0.25 + slat1 * 0.5), sn) * math.cos(slat1) / sn
        ro = re * sf / math.pow(math.tan(math.pi * 0.25 + olat * 0.5), sn)

        ra = re * sf / math.pow(math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5), sn)
        theta = lon * DEGRAD - olon

        if theta > math.pi:
            theta -= 2.0 * math.pi
        if theta < -math.pi:
            theta += 2.0 * math.pi
        theta *= sn

        x = int(ra * math.sin(theta) + self.XO + 0.5)
        y = int(ro - ra * math.cos(theta) + self.YO + 0.5)

        return x, y

    def _get_forecast_time(self) -> tuple[str, str]:
        """API 호출에 사용할 가장 최신의 예보 발표 시간을 계산합니다."""
        now = datetime.now()
        # 기상청 발표 시간: 02:10, 05:10, 08:10, 11:10, 14:10, 17:10, 20:10, 23:10
        # 안정성을 위해 40분 정도의 여유를 둠
        if now.minute < 40:
            now -= timedelta(hours=1)

        # 가장 가까운 과거의 발표 시간 찾기
        forecast_hours = [2, 5, 8, 11, 14, 17, 20, 23]
        base_hour = max([h for h in forecast_hours if h <= now.hour] or [23])

        if base_hour == 23 and now.hour < 2:
            now -= timedelta(days=1)

        base_date = now.strftime("%Y%m%d")
        base_time = f"{base_hour:02d}00"

        return base_date, base_time

    def _call_weather_api(
        self, nx: int, ny: int, base_date: str, base_time: str
    ) -> dict:
        """기상청 단기예보 API를 호출하고 응답을 반환합니다."""
        params = {
            "serviceKey": self.service_key,
            "pageNo": "1",
            "numOfRows": "1000",  # 충분한 수의 예보를 가져옴
            "dataType": "JSON",
            "base_date": base_date,
            "base_time": base_time,
            "nx": str(nx),
            "ny": str(ny),
        }

        try:
            response = requests.get(self.base_url, params=params, timeout=10)
            response.raise_for_status()  # 200 OK가 아니면 예외 발생

            data = response.json()
            header = data.get("response", {}).get("header", {})

            if header.get("resultCode") == "00":
                logger.info("API 호출 성공")
                return data
            else:
                error_msg = header.get("resultMsg", "Unknown API error")
                raise Exception(f"API Error: {error_msg}")
        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 실패: {e}")
            raise

    def _parse_weather_data(self, data: dict) -> dict:
        """API 응답 데이터를 파싱하여 필요한 날씨 정보만 추출합니다."""
        try:
            items = data["response"]["body"]["items"]["item"]

            # 현재 시간과 가장 가까운 예보 시간의 데이터만 필터링
            now = datetime.now()
            closest_forecast_time = min(
                (item for item in items if "fcstTime" in item),
                key=lambda x: abs(
                    now
                    - datetime.strptime(f"{x['fcstDate']}{x['fcstTime']}", "%Y%m%d%H%M")
                ),
            )["fcstTime"]

            weather_info = {}
            for item in items:
                if item["fcstTime"] == closest_forecast_time:
                    weather_info[item["category"]] = item["fcstValue"]

            # 날씨 상태 결정 (강수 형태 우선)
            pty_code = int(weather_info.get("PTY", 0))
            sky_code = int(weather_info.get("SKY", 1))
            condition = self._get_precipitation_type(
                pty_code
            ) or self._get_sky_condition(sky_code)

            return {
                "temperature": float(weather_info.get("TMP", 20.0)),
                "condition": condition,
                "humidity": int(weather_info.get("REH", 60)),
                "wind_speed": float(weather_info.get("WSD", 2.0)),
            }
        except (KeyError, IndexError, ValueError) as e:
            logger.error(f"데이터 파싱 실패: {e}")
            raise

    def _get_sky_condition(self, sky_code: int) -> str:
        """하늘 상태 코드를 문자열로 변환합니다."""
        return {1: "맑음", 3: "구름많음", 4: "흐림"}.get(sky_code, "알 수 없음")

    def _get_precipitation_type(self, pty_code: int) -> Optional[str]:
        """강수 형태 코드를 문자열로 변환합니다. 강수 없으면 None 반환."""
        return {1: "비", 2: "비/눈", 3: "눈", 4: "소나기"}.get(pty_code)

    def get_fallback_weather(self) -> dict:
        """API 호출 실패 시 사용할 기본 날씨 정보를 반환합니다."""
        logger.warning("폴백(기본) 날씨 정보를 사용합니다.")
        return {
            "temperature": 23.5,
            "condition": "맑음",
            "humidity": 60,
            "wind_speed": 5.2,
        }


# --- 테스트용 코드 ---
def test_weather_api():
    """날씨 API 클래스를 테스트하는 함수입니다."""
    weather_api = KoreaWeatherAPI()

    test_locations = [
        {"name": "서울", "lat": 37.5665, "lon": 126.9780},
        {"name": "부산", "lat": 35.1796, "lon": 129.0756},
        {"name": "제주", "lat": 33.4996, "lon": 126.5312},
    ]

    for location in test_locations:
        print(f"\n--- {location['name']} 날씨 조회 ---")
        weather = weather_api.get_weather_info(location["lat"], location["lon"])
        print(f"  - 온도: {weather['temperature']}°C")
        print(f"  - 상태: {weather['condition']}")
        print(f"  - 습도: {weather['humidity']}%")
        print(f"  - 풍속: {weather['wind_speed']}m/s")


if __name__ == "__main__":
    test_weather_api()
