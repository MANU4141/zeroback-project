#FLASK/BE import
from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import swag_from
from datetime import datetime
import json
import yaml
import os

#Ai modules import
import cv2
import numpy as np
from PIL import Image
import io

#swagger import
from swagger_config import init_swagger, swagger_template
from api_schemas import recommend_schema, health_check_schema, make_swagger_schema, debug_ai_status_schema, debug_weather_test_schema

#AI modules import
try:
    from ai.yolo_multitask import YOLOv11MultiTask
    from recommender.final_recommender import final_recommendation
    from config.config import CLASS_MAPPINGS, MODEL_PATHS
    from weather.recommend_by_weather import recommend_by_weather
    from llm.gemini_prompt_utils import analyze_user_prompt
    from weather_api import KoreaWeatherAPI  # 새로 추가
except ImportError as e:
    print(f"AI 모듈 import 실패: {e}")
    YOLOv11MultiTask = None

#Flask app init
app = Flask(__name__)
CORS(app)

#make Ai model global var
yolo_model=None
ai_model=None

# 날씨 API 인스턴스 생성
weather_api = KoreaWeatherAPI()

def combine_multiple_image_results(results_list):
    """여러 이미지 분석 결과를 통합하는 함수"""
    try:
        from collections import Counter
        
        # 유효한 결과만 필터링 (attributes가 있는 것만)
        valid_results = [r for r in results_list if r.get("attributes") is not None]
        
        if not valid_results:
            print("[BE] combine_multiple_image_results : no valid results to combine")
            return None
        
        print(f"[BE] combine_multiple_image_results : processing {len(valid_results)} valid results")
        
        combined_attributes = {}
        total_confidence = 0
        
        # 각 속성별로 결과 수집
        for result in valid_results:
            total_confidence += result.get("confidence", 0)
            attributes = result["attributes"]
            
            for attr_type, attr_values in attributes.items():
                if attr_type not in combined_attributes:
                    combined_attributes[attr_type] = []
                
                # 각 속성값들을 리스트에 추가
                if isinstance(attr_values, list):
                    for attr_value in attr_values:
                        if isinstance(attr_value, dict) and "class_name" in attr_value:
                            combined_attributes[attr_type].append(attr_value["class_name"])
                        else:
                            combined_attributes[attr_type].append(str(attr_value))
                else:
                    combined_attributes[attr_type].append(str(attr_values))
        
        # 각 속성별로 가장 많이 나온 값들 선택 (상위 3개)
        final_attributes = {}
        for attr_type, values in combined_attributes.items():
            if values:  # 빈 리스트가 아닌 경우만
                # 빈도수 계산
                counter = Counter(values)
                # 상위 3개 선택
                most_common = counter.most_common(3)
                final_attributes[attr_type] = [item[0] for item in most_common]
        
        print(f"[BE] combine_multiple_image_results : final attributes: {final_attributes}")
        
        return final_attributes
        
    except Exception as e:
        print(f"[BE] combine_multiple_image_results error: {e}")
        return None

def initialize_ai_models():
    global yolo_model, ai_model
    
    #class Yolov11MultiTask init check
    if YOLOv11MultiTask is None:
        print("Cant use Ai modules")
        return False
    
    try:
        #ultralytics->easy to use YOLOv11
        from ultralytics import YOLO
        import torch
        
        model_path = MODEL_PATHS.get("yolo")
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            num_classes_dict = {k: len(v) for k, v in CLASS_MAPPINGS.items()}
            device = "cuda" if torch.cuda.is_available() else "cpu"
            ai_model = YOLOv11MultiTask(yolo_model, num_classes_dict).to(device)
            
            print(f"AI 모델이 성공적으로 로드되었습니다. (Device: {device})")
            return True
        else:
            print(f"모델 파일을 찾을 수 없습니다: {model_path}")
            return False
            
    except Exception as e:
        print(f"AI 모델 초기화 실패: {e}")
        return False
    

_models_initialized = False
# 앱 시작 시 모델 초기화
@app.before_request
def startup():
    global _models_initialized
    if not _models_initialized:
        initialize_ai_models()
        _models_initialized = True

#swagger live access on /apidocs/
swagger=init_swagger(app)


#debug purpose only!!
@app.route('/api/makeswagger', methods=['GET'])
@swag_from(make_swagger_schema)
def make_swagger():
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_file_path = os.path.join(current_dir, 'swagger.yaml')
        
        # swagger_config의 template을 그대로 사용
        swagger_spec = swagger_template.copy()
        
        # 기존 스키마들을 paths에 추가
        swagger_spec['paths'] = {
            '/api/recommend': {
                'post': recommend_schema
            },
            '/api/health': {
                'get': health_check_schema
            },
            '/api/makeswagger': {
                'get': make_swagger_schema
            },
            '/api/debug/ai-status': {
                'get': debug_ai_status_schema
            },
            '/api/debug/weather-test': {
                'get': debug_weather_test_schema
            }
        }
        
        with open(yaml_file_path, 'w', encoding='utf-8') as f:
            yaml.dump(swagger_spec, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
        
        return jsonify({
            "message": "Swagger YAML file successfully created",
            "file_path": yaml_file_path,
            "timestamp": datetime.now().isoformat(),
            "file_size": f"{os.path.getsize(yaml_file_path)} bytes"
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Swagger file creation failed: {str(e)}",
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/recommend', methods=['POST'])
@swag_from(recommend_schema)
def recommend(): #content type -> multipart/form-data
    try:
        content_type=request.content_type
        if content_type and content_type.startswith('multipart/form-data'):
            #parse json data
            json_data=request.form.get('data')
            if not json_data:
                return jsonify({"error": "No data received"}), 400
            
            try:
                data=json.loads(json_data)
            except json.JSONDecodeError: #exception check
                return jsonify({"error": "Invalid JSON format"}), 400
            
            location=data.get('location')#should be a string, not coordinates
            latitude=data.get('latitude')#to use weather api
            longitude=data.get('longitude')
            style_select=data.get('style_select')#styles like '캐주얼','모던'...
            user_request=data.get('user_request')#user request strings

            uploaded_files=request.files.getlist('images') #handle multiple images
            
            # React와 Swagger 호환성을 위해 두 필드명 모두 확인
            if not uploaded_files:
                uploaded_files = request.files.getlist('image')  # 단수형도 확인
            if not uploaded_files:
                # 혹시 다른 필드명이 있는지 확인
                print(f"[BE] 사용 가능한 파일 필드들: {list(request.files.keys())}")
                uploaded_files = []
        else:
            return jsonify({"error":"Unsupported data format, use multipart/form-data"}), 400
        
        if not location:
            return jsonify({"error": "Location is required"}), 400
        if not latitude or not longitude:
            return jsonify({"error": "Latitude and longitude are required"}), 400
        if not style_select or len(style_select) == 0:
            return jsonify({"error": "Style selection is required"}), 400
        if not user_request or user_request.strip()=='':
            return jsonify({"error": "User request is required"}), 400
        
        # 이미지 분석 (있는 경우)
        ai_attributes=None

        print(f"[BE] 이미지 분석 조건 체크:")
        print(f"[BE] - uploaded_files 존재: {uploaded_files is not None}")
        print(f"[BE] - uploaded_files 개수: {len(uploaded_files) if uploaded_files else 0}")
        print(f"[BE] - ai_model 상태: {ai_model is not None}")
        
        if uploaded_files:
            for i, f in enumerate(uploaded_files):
                print(f"[BE] - 파일 {i}: {f.filename if f else 'None'}")

        if uploaded_files and len(uploaded_files)>0 and ai_model is not None:
            try:
                print("[BE] recommend process : uploaded image length:"+str(len(uploaded_files)))
                
                results_list=[]

                for idx, uploaded_file in enumerate(uploaded_files):
                    if uploaded_file.filename:
                        print("[BE] recommend process : processing image "+str(idx)+" / "+str(len(uploaded_files)-1))

                        try:
                            uploaded_file.seek(0)
                            image_bytes=uploaded_file.read()
                            image=Image.open(io.BytesIO(image_bytes))
                            image_np=cv2.cvtColor(np.array(image),cv2.COLOR_RGB2BGR)

                            print("[BE] recommend process : YOLO object detection start on "+str(idx))

                            results=ai_model.detect(image_np)
                            crops=ai_model.extract_crops(image_np, results)

                            #predict attributes
                            if crops and len(crops)>0:
                                print("[BE] recommend process : predict attributes on "+str(idx))
                                crop, bbox, conf, cls = crops[0]
                                
                                # 디버깅: 변수 타입 정보 출력
                                print(f"[BE] crop type: {type(crop)}, bbox type: {type(bbox)}, conf type: {type(conf)}, cls type: {type(cls)}")
                                print(f"[BE] bbox value: {bbox}")

                                image_attributes=ai_model.predict_attributes(crop, CLASS_MAPPINGS)

                                # bbox 안전하게 처리
                                try:
                                    if hasattr(bbox, 'tolist'):
                                        bbox_list = bbox.tolist()
                                    elif isinstance(bbox, (list, tuple)):
                                        bbox_list = list(bbox)
                                    else:
                                        bbox_list = [float(bbox)] if isinstance(bbox, (int, float)) else None
                                except Exception as bbox_error:
                                    print(f"[BE] bbox 변환 실패: {bbox_error}")
                                    bbox_list = None

                                image_result={
                                    'id': idx,
                                    'filename': uploaded_file.filename,
                                    "attributes": image_attributes,
                                    "confidence": float(conf) if conf is not None else 0.0,
                                    "detected_class": int(cls) if cls is not None else -1,
                                    "bbox": bbox_list
                                }
                                results_list.append(image_result)
                                print("[BE] recommend process : attributes prediction done on "+str(idx))
                            else:
                                print("[BE] recommend process : no valid crops found on "+str(idx))
                                image_result={
                                    'id': idx,
                                    'filename': uploaded_file.filename,
                                    "attributes": None,
                                    "confidence": 0.0,
                                    "detected_class": -1,
                                    "bbox": "No objects detected"
                                }
                                results_list.append(image_result)
                        except Exception as e:
                            print("[BE] recommend process : error occurred on "+str(idx)+": "+str(e))
                            error_result={
                                'id': idx,
                                'filename': uploaded_file.filename,
                                "error": str(e),
                                "confidence": 0.0,
                            }
                            results_list.append(error_result)
                
                if results_list:
                    print(f"[BE] recommend process : combining {len(results_list)} image results")
                    ai_attributes = combine_multiple_image_results(results_list)
                    print(f"[BE] recommend process : image analysis combination done")
                else:
                    print("[BE] recommend process : no valid image results")
                    ai_attributes = None
                               
            except Exception as e:
                print(f"[BE] 이미지 분석 실패: {e}")
                ai_attributes = None
        else:
            if not uploaded_files or len(uploaded_files) == 0:
                print("[BE] 업로드된 이미지가 없습니다")
            elif ai_model is None:
                print("[BE] AI 모델이 로드되지 않았습니다")
            else:
                print("[BE] 알 수 없는 이유로 이미지 분석을 건너뜁니다")
        
        # ✅ 실제 날씨 API 호출
        print(f"[WEATHER] 실제 날씨 조회 시작: 위도={latitude}, 경도={longitude}")
        try:
            weather_info = weather_api.get_weather_info(latitude, longitude)
            print(f"[WEATHER] 날씨 조회 성공: {weather_info}")
        except Exception as weather_error:
            print(f"[WEATHER] 날씨 API 실패: {weather_error}")
            # 폴백: 기본 날씨 정보 사용
            weather_info = {
                "temperature": 23.5,
                "condition": "맑음",
                "humidity": 60,
                "wind_speed": 5.2
            }

        # 최종 추천 생성
        try:
            recommendation_result = final_recommendation(
                weather=weather_info,
                user_prompt=user_request,
                style_preferences=style_select,
                ai_attributes=ai_attributes,
                gemini_api_key=None#os.getenv("GEMINI_API_KEY")
            )
            
            return jsonify({
                "success": True,
                "weather": weather_info,
                "recommendation_text": recommendation_result.get("recommendation_text", "추천을 생성했습니다."),
                "suggested_items": recommendation_result.get("categories", ["반팔티", "청바지"]),
                "ai_analysis": ai_attributes,
                "recommendation_details": recommendation_result
            }), 200
            
        except Exception as e:
            print(f"추천 생성 실패: {e}")
            # 폴백: 기존 더미 데이터 반환
            return jsonify({
                "success": True,
                "weather": weather_info,
                "recommendation_text": f"오늘 날씨에는 {', '.join(style_select)} 스타일로 {user_request}에 맞는 코디를 추천합니다.",
                "suggested_items": ["반팔티", "청바지", "스니커즈"],
                "ai_analysis": ai_attributes
            }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    

@app.route('/api/health', methods=['GET'])
@swag_from(health_check_schema)
def health_check():
    #check backend server status
    global ai_model, yolo_model
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'ai_model_loaded': ai_model is not None,
        'yolo_model_loaded': yolo_model is not None,
        'models_initialized': _models_initialized
    }), 200

@app.route('/api/debug/ai-status', methods=['GET'])
@swag_from(debug_ai_status_schema)
def debug_ai_status():
    """AI 모델 상태 디버깅용 엔드포인트"""
    global ai_model, yolo_model, _models_initialized
    
    status = {
        'models_initialized': _models_initialized,
        'ai_model_loaded': ai_model is not None,
        'yolo_model_loaded': yolo_model is not None,
        'YOLOv11MultiTask_available': YOLOv11MultiTask is not None,
    }
    
    try:
        from config.config import MODEL_PATHS
        model_path = MODEL_PATHS.get("yolo")
        status['model_path'] = model_path
        status['model_file_exists'] = os.path.exists(model_path) if model_path else False
    except Exception as e:
        status['config_error'] = str(e)
    
    return jsonify(status), 200

@app.route('/api/debug/weather-test', methods=['GET'])
@swag_from(debug_weather_test_schema)
def test_weather_api_endpoint():
    """날씨 API 테스트용 엔드포인트"""
    try:
        # URL 파라미터에서 좌표 가져오기
        lat = request.args.get('lat', 37.5665, type=float)  # 기본값: 서울
        lon = request.args.get('lon', 126.9780, type=float)
        
        print(f"[WEATHER_TEST] 테스트 요청: 위도={lat}, 경도={lon}")
        
        # 날씨 API 호출
        weather_data = weather_api.get_weather_info(lat, lon)
        
        return jsonify({
            'success': True,
            'location': {'latitude': lat, 'longitude': lon},
            'weather_data': weather_data,
            'timestamp': datetime.now().isoformat(),
            'api_key_status': 'OK' if weather_api.service_key_decoded else 'Missing'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500

if __name__ == '__main__':
    # AI 모델 사용 시 자동 리로드 비활성화 권장
    use_reloader = os.getenv('FLASK_USE_RELOADER', 'false').lower() == 'true'
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=use_reloader)