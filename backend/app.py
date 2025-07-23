from flask import Flask, request, jsonify
from flask_cors import CORS
from flasgger import swag_from
from datetime import datetime
import json
import yaml
import os

from swagger_config import init_swagger, swagger_template
from api_schemas import recommend_schema, health_check_schema, make_swagger_schema

app = Flask(__name__)
CORS(app)

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
        image_analysis = None
        if uploaded_file and uploaded_file.filename:
            # YOLO 이미지 분석 로직
            image_analysis = {
                "detected_items": ["상의", "하의"],
                "colors": ["파란색", "검은색"],
                "style_tags": ["캐주얼"]
            }
        
        #recommend return dummy data
        return jsonify({
            "success": True,
            "weather": {
                "temperature": 23.5,
                "condition": "맑음",
                "humidity": 60,
                "wind_speed": 5.2
            },
            "recommendation_text": f"오늘 날씨에는 {', '.join(style_select)} 스타일로 {user_request}에 맞는 코디를 추천합니다.",
            "suggested_items": ["반팔티", "청바지", "스니커즈"],
            "recommended_images": [
                {
                    "id": 1,
                    "category": "상의",
                    "item_name": "스트릿 반팔티",
                    "image_url": "/static/images/tshirt_1.jpg"
                }
            ],
            "image_analysis": image_analysis
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