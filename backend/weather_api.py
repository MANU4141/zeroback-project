"""
한국 기상청 공공데이터 API를 활용한 날씨 정보 조회 모듈
공공데이터포털 기상청_단기예보 조회서비스 사용
"""

import requests
import os
import math
from datetime import datetime, timedelta
from dotenv import load_dotenv

# 환경변수 로드
load_dotenv()

class KoreaWeatherAPI:
    def __init__(self):
        """기상청 API 클래스 초기화"""
        # 환경변수에서 API 키 로드
        self.service_key_encoded = os.getenv('WEATHER_API_KEY_ENCODE')
        self.service_key_decoded = os.getenv('WEATHER_API_KEY_DECODE')
        
        # API 기본 정보
        self.base_url = "http://apis.data.go.kr/1360000/VilageFcstInfoService_2.0"
        
        # 격자 변환 상수 (기상청 공식)
        self.RE = 6371.00877     # 지구 반경(km)
        self.GRID = 5.0          # 격자 간격(km)
        self.SLAT1 = 30.0        # 투영 위도1(degree)
        self.SLAT2 = 60.0        # 투영 위도2(degree)
        self.OLON = 126.0        # 기준점 경도(degree)
        self.OLAT = 38.0         # 기준점 위도(degree)
        self.XO = 43             # 기준점 X좌표(GRID)
        self.YO = 136            # 기준점 Y좌표(GRID)
        
        print(f"[WEATHER] API init", flush=True)
        print(f"[WEATHER] API key status : {'SET' if self.service_key_decoded else 'CANNOTREAD'}", flush=True)
    
    def get_weather_info(self, latitude, longitude):
        """
        위경도 좌표로 현재 날씨 정보 조회
        
        Args:
            latitude (float): 위도
            longitude (float): 경도
            
        Returns:
            dict: 날씨 정보
            {
                "temperature": 23.5,
                "condition": "맑음",
                "humidity": 60,
                "wind_speed": 5.2
            }
        """
        try:
            print(f"[WEATHER] REQ START: lat={latitude}, lon={longitude}", flush=True)
            
            # 1. 위경도를 기상청 격자좌표로 변환
            grid_x, grid_y = self.convert_coords_to_grid(latitude, longitude)
            print(f"[WEATHER] convert coordinates: X={grid_x}, Y={grid_y}", flush=True)
            
            # 2. API 호출용 시간 정보 생성
            base_date, base_time = self.get_forecast_time()
            print(f"[WEATHER] forecast base time : {base_date} {base_time}", flush=True)
            
            # 3. 기상청 API 호출
            weather_data = self.call_weather_api(grid_x, grid_y, base_date, base_time)
            
            # 4. 데이터 파싱 및 반환
            result = self.parse_weather_data(weather_data)
            print(f"[WEATHER] final result: {result}", flush=True)

            return result
            
        except Exception as e:
            print(f"[WEATHER] error : {e}", flush=True)
            return self.get_fallback_weather()
    
    def convert_coords_to_grid(self, lat, lon):
        """
        위경도를 기상청 격자좌표(X,Y)로 변환
        기상청 공식 알고리즘 사용
        """
        try:
            DEGRAD = math.pi / 180.0
            
            re = self.RE / self.GRID
            slat1 = self.SLAT1 * DEGRAD
            slat2 = self.SLAT2 * DEGRAD
            olon = self.OLON * DEGRAD
            olat = self.OLAT * DEGRAD
            
            sn = math.tan(math.pi * 0.25 + slat2 * 0.5) / math.tan(math.pi * 0.25 + slat1 * 0.5)
            sn = math.log(math.cos(slat1) / math.cos(slat2)) / math.log(sn)
            sf = math.tan(math.pi * 0.25 + slat1 * 0.5)
            sf = math.pow(sf, sn) * math.cos(slat1) / sn
            ro = math.tan(math.pi * 0.25 + olat * 0.5)
            ro = re * sf / math.pow(ro, sn)
            
            ra = math.tan(math.pi * 0.25 + lat * DEGRAD * 0.5)
            ra = re * sf / math.pow(ra, sn)
            theta = lon * DEGRAD - olon
            
            if theta > math.pi:
                theta -= 2.0 * math.pi
            if theta < -math.pi:
                theta += 2.0 * math.pi
            theta *= sn
            
            x = math.floor(ra * math.sin(theta) + self.XO + 0.5)
            y = math.floor(ro - ra * math.cos(theta) + self.YO + 0.5)
            
            return int(x), int(y)
            
        except Exception as e:
            print(f"[WEATHER] failed convert coordinates: {e}", flush=True)
            # 서울 기본 좌표 반환
            return 60, 127
    
    def get_forecast_time(self):
        """
        기상청 API 호출용 예보 시간 계산
        기상청은 하루 8번 특정 시간에만 예보 발표
        """
        now = datetime.now()
        
        # 기상청 발표 시간 (02, 05, 08, 11, 14, 17, 20, 23시)
        forecast_times = [
            "0200", "0500", "0800", "1100", 
            "1400", "1700", "2000", "2300"
        ]
        
        current_hour_min = now.strftime("%H%M")
        current_time = int(current_hour_min)
        
        # 현재 시간 이전의 가장 최근 발표 시간 찾기
        base_date = now.strftime("%Y%m%d")
        base_time = "0200"  # 기본값
        
        for time in reversed(forecast_times):
            if current_time >= int(time):
                base_time = time
                break
        else:
            # 오늘 발표 시간이 없으면 어제 마지막 시간
            yesterday = now - timedelta(days=1)
            base_date = yesterday.strftime("%Y%m%d")
            base_time = "2300"
        
        return base_date, base_time
    
    def call_weather_api(self, nx, ny, base_date, base_time):
        """기상청 단기예보 API 호출"""
        try:
            url = f"{self.base_url}/getVilageFcst"
            
            params = {
                'serviceKey': self.service_key_decoded,  # 디코딩된 키 사용
                'pageNo': '1',
                'numOfRows': '1000',
                'dataType': 'JSON',
                'base_date': base_date,
                'base_time': base_time,
                'nx': str(nx),
                'ny': str(ny)
            }
            
            print(f"[WEATHER] API call: {url}", flush=True)
            print(f"[WEATHER] param: nx={nx}, ny={ny}, date={base_date}, time={base_time}", flush=True)
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                # API 응답 상태 확인
                if data.get('response', {}).get('header', {}).get('resultCode') == '00':
                    print(f"[WEATHER] API call success", flush=True)
                    return data
                else:
                    error_msg = data.get('response', {}).get('header', {}).get('resultMsg', 'Unknown error')
                    print(f"[WEATHER] API error: {error_msg}", flush=True)
                    raise Exception(f"API Error: {error_msg}")
            else:
                print(f"[WEATHER] HTTP error: {response.status_code}", flush=True)
                raise Exception(f"HTTP Error: {response.status_code}")
                
        except Exception as e:
            print(f"[WEATHER] API call failed: {e}", flush=True)
            raise e
    
    def parse_weather_data(self, data):
        """
        기상청 API 응답 데이터를 파싱하여 필요한 정보 추출
        """
        try:
            items = data['response']['body']['items']['item']
            
            # 현재 시간과 가장 가까운 예보 데이터 찾기
            current_time = datetime.now()
            target_forecast = None
            min_time_diff = float('inf')
            
            # 예보 데이터 중 현재 시간과 가장 가까운 것 선택
            for item in items:
                fcst_date = item['fcstDate']
                fcst_time = item['fcstTime']
                
                try:
                    forecast_datetime = datetime.strptime(f"{fcst_date}{fcst_time}", "%Y%m%d%H%M")
                    time_diff = abs((forecast_datetime - current_time).total_seconds())
                    
                    if time_diff < min_time_diff:
                        min_time_diff = time_diff
                        target_forecast = f"{fcst_date}{fcst_time}"
                except:
                    continue
            
            # 선택된 시간의 모든 기상 요소 수집
            weather_info = {
                'temperature': None,    # TMP: 온도
                'humidity': None,       # REH: 습도
                'wind_speed': None,     # WSD: 풍속
                'condition': '맑음',    # SKY: 하늘상태, PTY: 강수형태
                'rain_prob': 0,         # POP: 강수확률
                'precipitation': 0      # PCP: 강수량
            }
            
            for item in items:
                fcst_date = item['fcstDate']
                fcst_time = item['fcstTime']
                current_forecast = f"{fcst_date}{fcst_time}"
                
                # 선택된 예보 시간의 데이터만 처리
                if current_forecast == target_forecast:
                    category = item['category']
                    value = item['fcstValue']
                    
                    try:
                        if category == 'TMP':  # 온도(℃)
                            weather_info['temperature'] = float(value)
                        elif category == 'REH':  # 습도(%)
                            weather_info['humidity'] = int(value)
                        elif category == 'WSD':  # 풍속(m/s)
                            weather_info['wind_speed'] = float(value)
                        elif category == 'POP':  # 강수확률(%)
                            weather_info['rain_prob'] = int(value)
                        elif category == 'SKY':  # 하늘상태
                            sky_code = int(value)
                            weather_info['condition'] = self.get_sky_condition(sky_code)
                        elif category == 'PTY':  # 강수형태
                            pty_code = int(value)
                            if pty_code > 0:
                                weather_info['condition'] = self.get_precipitation_type(pty_code)
                    except ValueError:
                        continue
            
            # 최종 결과 구성
            result = {
                "temperature": weather_info['temperature'] or 20.0,
                "condition": weather_info['condition'],
                "humidity": weather_info['humidity'] or 60,
                "wind_speed": weather_info['wind_speed'] or 2.0
            }
            
            return result
            
        except Exception as e:
            print(f"[WEATHER] failed parse data: {e}", flush=True)
            return self.get_fallback_weather()
    
    def get_sky_condition(self, sky_code):
        """하늘상태 코드를 문자열로 변환"""
        sky_conditions = {
            1: "맑음",
            3: "구름많음",
            4: "흐림"
        }
        return sky_conditions.get(sky_code, "맑음")
    
    def get_precipitation_type(self, pty_code):
        """강수형태 코드를 문자열로 변환"""
        precipitation_types = {
            1: "비",
            2: "비/눈",
            3: "눈",
            4: "소나기"
        }
        return precipitation_types.get(pty_code, "비")
    
    def get_fallback_weather(self):
        """API 호출 실패 시 기본 날씨 정보 반환"""
        return {
            "temperature": 23.5,
            "condition": "맑음", 
            "humidity": 60,
            "wind_speed": 5.2
        }


# 테스트 및 사용 예시
def test_weather_api():
    """날씨 API 테스트 함수"""
    weather_api = KoreaWeatherAPI()
    
    # 테스트 좌표들
    test_locations = [
        {"name": "서울", "lat": 37.5665, "lon": 126.9780},
        {"name": "부산", "lat": 35.1796, "lon": 129.0756},
        {"name": "대구", "lat": 35.8714, "lon": 128.6014},
        {"name": "인천", "lat": 37.4563, "lon": 126.7052}
    ]
    
    for location in test_locations:
        print(f"\n{'='*50}", flush=True)
        print(f"{location['name']} 날씨 조회", flush=True)
        print(f"{'='*50}", flush=True)
        
        weather = weather_api.get_weather_info(location['lat'], location['lon'])
        
        print(f"온도: {weather['temperature']}°C", flush=True)
        print(f"날씨: {weather['condition']}", flush=True)
        print(f"습도: {weather['humidity']}%", flush=True)
        print(f"풍속: {weather['wind_speed']}m/s", flush=True)


if __name__ == "__main__":
    print("한국 기상청 날씨 API 테스트", flush=True)
    test_weather_api()
