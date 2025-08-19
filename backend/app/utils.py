# backend/app/utils.py
"""
백엔드 유틸리티 함수들 (프로토타입 단순화)
- 이미지 처리 및 검증
- DB 구축
- 입력 검증
"""
import io
import logging
import os
import json
from collections import Counter
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from flask import request
from PIL import Image

from config import Config

logger = logging.getLogger(__name__)


def _extract_labels_from_data(label_data: Dict) -> Dict[str, str]:
    """라벨 JSON에서 스타일/색상/소재/카테고리/디테일 추출"""
    out = {
        "style": set(),
        "color": set(),
        "material": set(),
        "category": set(),
        "detail": set(),
    }
    labeling = (
        label_data.get("데이터셋 정보", {})
        .get("데이터셋 상세설명", {})
        .get("라벨링", {})
    )

    # 스타일
    for s in labeling.get("스타일", []):
        if isinstance(s, dict):
            if s.get("스타일"):
                out["style"].add(s["스타일"])
            if s.get("서브스타일"):
                out["style"].add(s["서브스타일"])

    # 의류 파트
    for part in ["아우터", "상의", "하의", "원피스"]:
        for item in labeling.get(part, []):
            if not isinstance(item, dict):
                continue
            if item.get("색상"):
                out["color"].add(item["색상"])
            if item.get("카테고리"):
                out["category"].add(item["카테고리"])

            for key in ["소재", "디테일"]:
                vals = item.get(key)
                if not vals:
                    continue
                k = "material" if key == "소재" else "detail"
                if isinstance(vals, list):
                    for v in vals:
                        if v:
                            out[k].add(v)
                else:
                    out[k].add(vals)

    # set -> 정렬된 문자열
    return {k: ", ".join(sorted(v)) if v else "" for k, v in out.items()}


def build_db_images(labels_dir: str, image_dir: str) -> List[Dict]:
    """라벨/이미지 디렉토리에서 DB용 메타 데이터 생성"""
    db_images: List[Dict] = []
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

    for jf in json_files:
        base = os.path.splitext(jf)[0]
        matched_img = next(
            (
                base + ext
                for ext in [".jpg", ".jpeg", ".png"]
                if base + ext in image_files
            ),
            None,
        )
        if not matched_img:
            continue

        image_path = os.path.join(image_dir, matched_img)
        label_path = os.path.join(labels_dir, jf)

        try:
            with open(label_path, "r", encoding="utf-8") as f:
                label_data = json.load(f)
            label_info = _extract_labels_from_data(label_data)
            db_images.append({"img_path": image_path, "label": label_info})
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            logger.warning(f"JSON 파싱/디코딩 실패 - {label_path}: {e}")
        except Exception as e:
            logger.error(f"DB 이미지 빌드 예외 ({label_path}): {e}")

    logger.info(f"최종 DB 이미지 개수: {len(db_images)}")
    return db_images


def parse_images(field_names=("images", "image")) -> List:
    """Request에서 이미지 파일 목록을 파싱/검증"""
    files = []
    for name in field_names:
        fl = request.files.getlist(name)
        if fl:
            files = fl
            break

    if not files:
        return []

    # 파일 개수 제한
    max_images = Config.INPUT_VALIDATION["max_images_per_request"]
    if len(files) > max_images:
        raise ValueError(
            f"최대 {max_images}개의 이미지만 업로드 가능합니다. (현재: {len(files)}개)"
        )

    # 파일별 기본 검증 (파일명/크기)
    validated = []
    max_size_bytes = Config.INPUT_VALIDATION["max_file_size_mb"] * 1024 * 1024
    for file in files:
        if not file.filename:
            raise ValueError("파일명이 없는 파일이 포함되어 있습니다.")

        # Flask의 MAX_CONTENT_LENGTH로 1차 방어 권장(앱 레벨 설정)
        file.seek(0, 2)
        size = file.tell()
        file.seek(0)

        if size > max_size_bytes:
            raise ValueError(
                f"파일 크기가 너무 큽니다: {file.filename} "
                f"({size / (1024*1024):.1f}MB > {Config.INPUT_VALIDATION['max_file_size_mb']}MB)"
            )
        validated.append(file)
    return validated


def preprocess_image(uploaded_file) -> np.ndarray:
    """업로드 이미지를 전처리: MIME 검증, 해상도 체크, 필요시 다운스케일, cv2 BGR 변환"""
    if not uploaded_file:
        raise ValueError("이미지 파일이 제공되지 않았습니다.")

    allowed = Config.INPUT_VALIDATION["allowed_mimetypes"]
    if uploaded_file.mimetype not in allowed:
        raise ValueError(
            f"지원되지 않는 이미지 형식입니다. 지원 형식: {', '.join(allowed)}"
        )

    try:
        uploaded_file.seek(0)
        data = uploaded_file.read()
        if not data:
            raise ValueError("빈 파일입니다.")
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        raise ValueError(f"이미지 파일이 손상되었거나 읽을 수 없습니다: {str(e)}")

    w, h = image.size
    min_res = Config.INPUT_VALIDATION["min_resolution"]
    if w < min_res or h < min_res:
        raise ValueError(
            f"이미지 해상도가 너무 낮습니다. 최소 {min_res}x{min_res} 필요합니다. (현재: {w}x{h})"
        )

    max_res = 2048
    if w > max_res or h > max_res:
        ratio = min(max_res / w, max_res / h)
        nw, nh = int(w * ratio), int(h * ratio)
        image = image.resize((nw, nh), Image.Resampling.LANCZOS)
        logger.debug(f"이미지 크기 조정: {w}x{h} -> {nw}x{nh}")

    img_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    return img_np


def validate_coordinates(latitude, longitude) -> Tuple[float, float]:
    """위도/경도 범위 검증"""
    bounds = Config.INPUT_VALIDATION["coordinate_bounds"]
    try:
        lat = float(latitude)
        lng = float(longitude)
    except (ValueError, TypeError):
        raise ValueError("위도와 경도는 숫자여야 합니다.")

    if not (bounds["lat_min"] <= lat <= bounds["lat_max"]):
        raise ValueError(
            f"위도가 유효 범위를 벗어났습니다. ({bounds['lat_min']} ~ {bounds['lat_max']})"
        )
    if not (bounds["lng_min"] <= lng <= bounds["lng_max"]):
        raise ValueError(
            f"경도가 유효 범위를 벗어났습니다. ({bounds['lng_min']} ~ {bounds['lng_max']})"
        )
    return lat, lng


def combine_multiple_image_results(image_results: List[Dict]) -> Dict:
    """여러 이미지의 AI 분석 결과를 빈도/요약으로 통합"""
    if not image_results:
        return {
            "combined_categories": [],
            "combined_colors": [],
            "combined_styles": [],
            "confidence_summary": {"avg": 0, "min": 0, "max": 0},
            "image_count": 0,
        }

    all_categories, all_colors, all_styles, all_confs = [], [], [], []

    for res in image_results:
        # YOLO
        for det in res.get("yolo_results", []) or []:
            cat = det.get("category")
            conf = det.get("confidence")
            if cat:
                all_categories.append(cat)
            if conf is not None:
                all_confs.append(conf)
        # ResNet
        r = res.get("resnet_results") or {}
        if isinstance(r, dict):
            if r.get("color"):
                all_colors.append(r["color"])
            if r.get("style"):
                all_styles.append(r["style"])

    cat_counts = Counter(all_categories)
    color_counts = Counter(all_colors)
    style_counts = Counter(all_styles)

    conf_summary = {
        "avg": sum(all_confs) / len(all_confs) if all_confs else 0,
        "min": min(all_confs) if all_confs else 0,
        "max": max(all_confs) if all_confs else 0,
    }

    return {
        "combined_categories": [k for k, _ in cat_counts.most_common(5)],
        "combined_colors": [k for k, _ in color_counts.most_common(3)],
        "combined_styles": [k for k, _ in style_counts.most_common(3)],
        "confidence_summary": conf_summary,
        "image_count": len(image_results),
        "detailed_counts": {
            "categories": dict(cat_counts),
            "colors": dict(color_counts),
            "styles": dict(style_counts),
        },
    }


def convert_image_paths_to_filenames(images_list: List[Dict]) -> List[Dict]:
    """img_path를 파일명으로 축약"""
    if not images_list:
        return images_list
    out = []
    for img in images_list:
        if isinstance(img, dict) and "img_path" in img:
            cp = img.copy()
            cp["img_path"] = os.path.basename(cp["img_path"])
            out.append(cp)
        else:
            out.append(img)
    return out
