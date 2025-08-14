from fastapi import FastAPI, UploadFile, Form
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from config.config import CLASS_MAPPINGS, MODEL_PATHS
from AI.yolo_multitask import YOLOv11MultiTask
from recommender.final_recommender import final_recommendation

app = FastAPI()
device = "cuda" if torch.cuda.is_available() else "cpu"
yolo_model = YOLO(MODEL_PATHS["yolo"])
num_classes_dict = {k: len(v) for k, v in CLASS_MAPPINGS.items()}
model = YOLOv11MultiTask(yolo_model, num_classes_dict).to(device)


@app.post("/api/recommend/")
async def recommend_api(
    file: UploadFile,
    temperature: float = Form(...),
    condition: str = Form(...),
    user_prompt: str = Form(...),
    style_preferences: str = Form(...),
):
    img_arr = np.frombuffer(await file.read(), np.uint8)
    image = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
    results = model.detect(image)
    crops_raw = model.extract_crops(image, results)
    ai_attributes = {}
    if crops_raw:
        ai_attributes = model.predict_attributes(
            crops_raw[0][0], CLASS_MAPPINGS, device=device
        )
        ai_attributes = {
            k: [v["class_name"] for v in preds] for k, preds in ai_attributes.items()
        }
    weather = {"temperature": temperature, "condition": condition}
    style_prefs = [
        pref.strip() for pref in style_preferences.split(",") if pref.strip()
    ]
    recommendation = final_recommendation(
        weather=weather,
        user_prompt=user_prompt,
        style_preferences=style_prefs,
        ai_attributes=ai_attributes,
        gemini_api_key=None,
    )
    return recommendation
