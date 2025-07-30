import os, sys
import torch
import cv2
from ultralytics import YOLO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config.config import CLASS_MAPPINGS, MODEL_PATHS
from AI.yolo_multitask import YOLOv11MultiTask


def test_yolo_multitask_prediction():
    yolo_model = YOLO(MODEL_PATHS["yolo"])
    num_classes_dict = {k: len(v) for k, v in CLASS_MAPPINGS.items()}
    model = YOLOv11MultiTask(yolo_model, num_classes_dict)
    image_path = r"C:\Users\rkdgu\Desktop\github_zeroback\zeroback-project\Algorithm\AI\images\test.jpg"  # 실제 경로/샘플 이미지 필요
    image = cv2.imread(image_path)
    assert image is not None, "이미지 파일을 찾을 수 없습니다."
    results = model.detect(image)
    crops = model.extract_crops(image, results)
    assert isinstance(crops, list) and len(crops) >= 1
    crop, bbox, conf, cls = crops[0]
    attrs = model.predict_attributes(crop, CLASS_MAPPINGS)
    print("탐지 박스:", bbox)
    print("예측 속성:", attrs)
    assert "category" in attrs


if __name__ == "__main__":
    test_yolo_multitask_prediction()
    print("AI 멀티태스킹 테스트 OK")
