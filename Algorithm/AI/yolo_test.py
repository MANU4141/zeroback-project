import sys
import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import platform

# ========================================
# 1. 프로젝트 루트(sys.path) 추가 (AI 폴더에서 파생)
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(os.path.join(os.path.dirname(current_file), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from recommender.final_recommender import final_recommendation
from config.config import CLASS_MAPPINGS, MODEL_PATHS
from AI.yolo_multitask import YOLOv11MultiTask


# ========================================
# 2. 한글폰트 자동 경로 탐색
def get_korean_font():
    if platform.system() == "Windows":
        path = "C:/Windows/Fonts/malgun.ttf"
    elif platform.system() == "Darwin":
        path = "/Library/Fonts/AppleGothic.ttf"
    else:
        path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if not os.path.isfile(path):
        print(f"[경고] 한글폰트 파일을 찾지 못했습니다: {path}. 기본폰트로 대체됩니다.")
    return path


FONT_PATH = get_korean_font()


# ========================================
# 3. 바운딩박스 & 한글·속성 시각화 (Pillow)
def generate_class_colors(class_names):
    base_colors = [
        (31, 119, 180),
        (255, 127, 14),
        (44, 160, 44),
        (214, 39, 40),
        (148, 103, 189),
        (140, 86, 75),
        (227, 119, 194),
        (127, 127, 127),
        (188, 189, 34),
        (23, 190, 207),
        (174, 199, 232),
        (255, 187, 120),
        (152, 223, 138),
        (255, 152, 150),
        (197, 176, 213),
        (196, 156, 148),
        (247, 182, 210),
        (199, 199, 199),
        (219, 219, 141),
        (158, 218, 229),
    ]
    return {cls: base_colors[i % len(base_colors)] for i, cls in enumerate(class_names)}


def draw_boxes_and_labels_pil(image, boxes_info, font_path=FONT_PATH):
    img_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    class_names = list({info["label"] for info in boxes_info})
    color_map = generate_class_colors(class_names)
    try:
        font = ImageFont.truetype(font_path, size=22)
        small_font = ImageFont.truetype(font_path, size=15)
    except Exception:
        font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    for info in boxes_info:
        x1, y1, x2, y2 = info["bbox"]
        color = color_map[info["label"]]
        # 박스(테두리)
        draw.rectangle([(x1, y1), (x2, y2)], outline=color, width=3)
        # 메인 라벨
        label_txt = f"{info['label']} {info['conf']*100:.1f}%"
        # 세부 속성(아래에 작게)
        details_txt = info.get("details_str", None)
        # 메인라벨 크기 계산 (textbbox)
        bbox = draw.textbbox((0, 0), label_txt, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        # 배경라벨(반투명 박스)
        text_y = max(0, y1 - th - 12)
        draw.rectangle(
            [(x1, text_y), (x1 + tw + 8, text_y + th + 8)],
            fill=tuple(list(color) + [192]),
        )
        draw.text((x1 + 4, text_y + 4), label_txt, font=font, fill=(255, 255, 255))
        # 세부라벨(아래쪽)
        if details_txt:
            sbbox = draw.textbbox((0, 0), details_txt, font=small_font)
            stw, sth = sbbox[2] - sbbox[0], sbbox[3] - sbbox[1]
            draw.rectangle(
                [(x1, y2), (x1 + stw + 8, y2 + sth + 10)],
                fill=tuple(list(color) + [192]),
            )
            draw.text(
                (x1 + 4, y2 + 4), details_txt, font=small_font, fill=(255, 255, 255)
            )
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)


# ========================================
# 4. 메인 동작
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    yolo_pt_path = MODEL_PATHS["yolo"]
    num_classes_dict = {task: len(classes) for task, classes in CLASS_MAPPINGS.items()}
    yolo_model = YOLO(yolo_pt_path)
    model = YOLOv11MultiTask(yolo_model)
    from AI.resnet_multitask import FashionAttributePredictor

    resnet_model = FashionAttributePredictor(device=device)

    image_path = os.path.join(
        os.path.dirname(__file__),
        R"D:\zeroback_KHJ_end\zeroback-project\backend\test.png",  # 테스트 이미지 경로
    )
    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return

    results = model.detect(image)
    crops_raw = model.extract_crops(image, results)
    print(f"탐지된 객체 수: {len(crops_raw)}")

    weather = {"temperature": 25, "condition": "맑음"}
    user_prompt = "오늘 데이트라서 귀엽고 깔끔하게 입고 싶어요. 파스텔톤 좋아요."
    style_preferences = ["로맨틱", "캐주얼"]

    boxes_info = []
    for idx, (crop, bbox, conf, cls_id) in enumerate(crops_raw):
        # 카테고리명 가져오기
        cls_name = (
            CLASS_MAPPINGS["category"][cls_id]
            if "category" in CLASS_MAPPINGS and cls_id < len(CLASS_MAPPINGS["category"])
            else f"cls{cls_id}"
        )
        # 세부 속성 추출 및 print (ResNet 사용)
        ai_attributes = resnet_model.predict_attributes(crop, CLASS_MAPPINGS)
        print(f"\n[{idx+1}] bbox: {bbox}, conf: {conf:.4f}, cls: {cls_id} ({cls_name})")
        # 주요 속성만 아래쪽에 라벨로 요약
        detail_keys = ["color", "fit", "style", "material", "print", "detail", "collar"]
        details_str = " | ".join(
            [
                f"{k}:{','.join([p['class_name'] for p in ai_attributes.get(k, [])[:2]])}"
                for k in detail_keys
                if ai_attributes.get(k)
            ]
        )
        boxes_info.append(
            {"bbox": bbox, "conf": conf, "label": cls_name, "details_str": details_str}
        )
        for task, preds in ai_attributes.items():
            print(f"  {task}:")
            for v in preds:
                print(f"    - {v['class_name']} ({v['probability']:.4f})")
        simple_attrs = {
            task: [p["class_name"] for p in preds]
            for task, preds in ai_attributes.items()
        }
        recommendation = final_recommendation(
            weather=weather,
            user_prompt=user_prompt,
            style_preferences=style_preferences,
            ai_attributes=simple_attrs,
            gemini_api_key=None,
        )
        print("=== 최종 추천 결과 ===")
        print(recommendation["recommendation_text"])
        print("추천 아이템:", recommendation["suggested_items"])

    # --- 시각화 및 저장 ---
    img_out = draw_boxes_and_labels_pil(image, boxes_info, font_path=FONT_PATH)
    out_path = os.path.join(
        os.path.dirname(__file__), "images", "pred_result.jpg"  # 예측 결과 저장 경로
    )
    cv2.imwrite(out_path, img_out)
    print(f"\n결과 시각화 이미지를 저장했습니다: {out_path}")


if __name__ == "__main__":
    main()
