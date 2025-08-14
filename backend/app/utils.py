# backend/app/utils.py
import os
import json
import logging
import io
from flask import request
from PIL import Image
import numpy as np
import cv2
from collections import Counter

logger = logging.getLogger(__name__)


def _extract_labels_from_data(label_data):
    """라벨 데이터에서 스타일, 색상, 소재 등 속성을 추출하는 헬퍼 함수"""
    label_info = {
        "style": set(),
        "color": set(),
        "material": set(),
        "category": set(),
        "detail": set(),
    }
    라벨링 = (
        label_data.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("라벨링", {})
    )

    for s in 라벨링.get("스타일", []):
        if isinstance(s, dict):
            if s.get("스타일"):
                label_info["style"].add(s["스타일"])
            if s.get("서브스타일"):
                label_info["style"].add(s["서브스타일"])

    for part in ["아우터", "상의", "하의", "원피스"]:
        for item in 라벨링.get(part, []):
            if not isinstance(item, dict):
                continue
            if item.get("색상"):
                label_info["color"].add(item["색상"])
            if item.get("카테고리"):
                label_info["category"].add(item["카테고리"])

            for key in ["소재", "디테일"]:
                values = item.get(key)
                if values:
                    english_key = "material" if key == "소재" else "detail"
                    if isinstance(values, list):
                        for v in values:
                            if v:
                                label_info[english_key].add(v)
                    else:
                        label_info[english_key].add(values)

    for k, v_set in label_info.items():
        label_info[k] = ", ".join(sorted(list(v_set))) if v_set else ""
    return label_info


def build_db_images(labels_dir, image_dir):
    """지정된 디렉토리에서 이미지와 라벨 데이터를 로드하여 DB를 구축합니다."""
    db_images = []
    logger.info(f"DB 이미지 빌드 시작: 라벨='{labels_dir}', 이미지='{image_dir}'")

    try:
        json_files = [f for f in os.listdir(labels_dir) if f.lower().endswith(".json")]
        image_files = {
            f
            for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        }
    except FileNotFoundError as e:
        logger.error(f"DB 이미지/라벨 디렉토리 없음: {e}")
        return []

    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]
        matched_img = next(
            (
                f
                for ext in [".jpg", ".jpeg", ".png"]
                if (f := base_name + ext) in image_files
            ),
            None,
        )

        if not matched_img:
            continue

        image_path = os.path.join(image_dir, matched_img)
        label_path = os.path.join(labels_dir, json_file)

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            label_info = _extract_labels_from_data(label_data)
            db_images.append({"img_path": image_path, "label": label_info})
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"JSON 파싱/디코딩 실패 - {label_path}: {e}")
        except Exception as e:
            logger.error(f"DB 이미지 빌드 중 예외 발생 ({label_path}): {e}")

    logger.info(f"최종 DB 이미지 개수: {len(db_images)}")
    return db_images


def parse_images(field_names=("images", "image")):
    """Request에서 이미지 파일 목록을 파싱합니다."""
    for name in field_names:
        files = request.files.getlist(name)
        if files:
            return files
    return []


def combine_multiple_image_results(results_list):
    """여러 이미지 분석 결과를 통합하여 요약합니다."""
    valid_results = [r for r in results_list if r.get("attributes") is not None]
    if not valid_results:
        return None

    combined = {"category": [], "style": [], "color": [], "material": [], "detail": []}
    for r in valid_results:
        if r.get("category"):
            combined["category"].append(r["category"])
        for k, v in r.get("attributes", {}).items():
            if k in combined:
                items_to_add = v if isinstance(v, list) else [v]
                for item in items_to_add:
                    class_name = (
                        item.get("class_name") if isinstance(item, dict) else str(item)
                    )
                    combined[k].append(class_name)

    summary = {}
    for k, vlist in combined.items():
        if vlist:
            counter = Counter(vlist)
            summary[k] = [item for item, _ in counter.most_common(3)]
    return summary


def preprocess_image(uploaded_file):
    """업로드된 파일을 이미지 객체로 전처리합니다."""
    if not uploaded_file.filename:
        raise ValueError("파일명이 없습니다.")
    if uploaded_file.mimetype not in ["image/jpeg", "image/png"]:
        raise ValueError("지원되지 않는 이미지 형식입니다.")

    uploaded_file.seek(0)
    image_bytes = uploaded_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return image_np
