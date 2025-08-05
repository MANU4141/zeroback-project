from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import swag_from
from datetime import datetime
import json
import yaml
import os

import cv2
import numpy as np
from PIL import Image
import io

from swagger_config import init_swagger, swagger_template
from api_schemas import recommend_schema, health_check_schema, make_swagger_schema

try:
    from ai.yolo_multitask import YOLOv11MultiTask
    from recommender.final_recommender import final_recommendation
    from config.config import CLASS_MAPPINGS, MODEL_PATHS
    from weather.recommend_by_weather import recommend_by_weather
    from llm.gemini_prompt_utils import analyze_user_prompt
except ImportError as e:
    print(f"AI 모듈 import 실패: {e}")
    YOLOv11MultiTask = None

app = Flask(__name__)
CORS(app)

yolo_model=None
ai_model=None

def initialize_ai_models():
    global yolo_model, ai_model
    
    if YOLOv11MultiTask is None:
        print("AI 모듈을 사용할 수 없습니다.")
        return False
    
    try:
        from ultralytics import YOLO
        import torch
        
        # YOLO 모델 로드
        model_path = MODEL_PATHS.get("yolo")
        if os.path.exists(model_path):
            yolo_model = YOLO(model_path)
            
            # 멀티태스크 모델 초기화
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
            style_select=data.get('style_select')#styles like '캐주얼','모던'...
            user_request=data.get('user_request')#user request strings

            uploaded_file=request.files.get('image')
        else:
            return jsonify({"error":"Unsupported data format, use multipart/form-data"}), 400
        
        if not location:
            return jsonify({"error": "Location is required"}), 400
        if not style_select or len(style_select) == 0:
            return jsonify({"error": "Style selection is required"}), 400
        if not user_request or user_request.strip()=='':
            return jsonify({"error": "User request is required"}), 400
        
        # 이미지 분석 (있는 경우)
        ai_attrib=None
        if uploaded_file and uploaded_file.filename and ai_model is not None:
            try:
                # 이미지 전처리
                image_bytes = uploaded_file.read()
                image = Image.open(io.BytesIO(image_bytes))
                image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
                
                # YOLO 객체 탐지
                results = ai_model.detect(image_np)
                crops = ai_model.extract_crops(image_np, results)
                
                # 속성 예측
                if crops:
                    crop, bbox, conf, cls = crops[0]  # 첫 번째 객체만 사용
                    ai_attributes = ai_model.predict_attributes(crop, CLASS_MAPPINGS)
                    
            except Exception as e:
                print(f"이미지 분석 실패: {e}")
                ai_attributes = None
        
        # 날씨 정보 (더미 데이터 - 실제로는 날씨 API 호출)
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
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat()
    }), 200

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)