from flasgger import Swagger

swagger_config={
    "headers":[],
    "specs":[
        {"endpoint":'apispec_1',
         "route":'/apispec_1.json',
         "rule_filter":lambda rule: True,
         "model_filter":lambda tag: True,
         }
    ],
    "static_url_path":"/flasgger_static",
    "swagger_ui":True,
    "specs_route":"/apidocs/"
}

swagger_template={
    "swagger": "2.0",
    "info": {
        "title": "OOTD-AI API DOCS",
        "description": "AI 기반 의상 추천 서비스 API",
        "version": "1.0.0"
    },
    "host": "localhost:5000",
    "basePath": "/",
    "schemes": ["http"],
    "consumes": ["application/json", "multipart/form-data"],
    "produces": ["application/json"],
    "definitions": {
        "Error": {
            "type": "object",
            "properties": {
                "error": {
                    "type": "string",
                    "description": "오류 메시지"
                }
            },
            "required": ["error"]
        }
    }
}

def init_swagger(app):
    return Swagger(app,config=swagger_config, template=swagger_template)