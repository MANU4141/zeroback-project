import os
import sys
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics, box_iou
from collections import defaultdict, Counter
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    top_k_accuracy_score,
    balanced_accuracy_score,
)
import pandas as pd
from datetime import datetime
import glob
from tqdm.auto import tqdm
import warnings

warnings.filterwarnings("ignore")

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# 설정 불러오기
from config.config import CLASS_MAPPINGS

# 기본 설정
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = r"D:\zeroback_KHJ_end\zeroback-project\backend\DATA\images"
LABEL_DIR = r"D:\zeroback_KHJ_end\zeroback-project\backend\DATA\labels"
MODEL_PATH = r"D:\zeroback_KHJ_end\zeroback-project\backend\models\YOLOv11_large.pt"
RESULTS_DIR = r"D:\zeroback_KHJ_end\zeroback-project\AI\YOLOv11_summary\RESULTS"

# 21개 카테고리 정의
CATEGORIES = CLASS_MAPPINGS["category"]
NUM_CLASSES = len(CATEGORIES)

# 카테고리 매핑: 세부 카테고리 → 대분류
CATEGORY_GROUPS = {
    "상의": ["탑", "블라우스", "티셔츠", "니트웨어", "셔츠", "브라탑", "후드티"],
    "하의": ["청바지", "팬츠", "스커트", "레깅스", "조거팬츠"],
    "아우터": ["코트", "재킷", "점퍼", "패딩", "베스트", "가디건", "짚업"],
    "원피스": ["드레스", "점프수트"],
}

# 한글→영어 키 맵핑
KOR_TO_ENG = {
    "카테고리": "category",
    "색상": "color",
    "소매기장": "sleeve_length",
    "기장": "sleeve_length",
    "옷깃": "collar",
    "넥라인": "neckline",
    "핏": "fit",
    "프린트": "print",
    "소재": "material",
    "디테일": "detail",
    "스타일": "style",
}


class YOLODetectionValidator:
    """YOLO 객체 탐지 + 분류 성능 검증 클래스 (실제 mAP 포함)"""

    def __init__(self, model_path, device=None):
        self.device = device or DEVICE
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        # YOLO 모델 로드
        self.model = YOLO(model_path)
        print(f"✅ YOLO 모델 로드 완료: {model_path}")
        print(f"📱 사용 디바이스: {self.device}")

        # mAP 계산을 위한 메트릭 초기화
        self.det_metrics = DetMetrics()

    def extract_bbox_from_json(self, json_data, img_width=800, img_height=800):
        """JSON에서 바운딩 박스 정보 추출"""
        bboxes = []

        try:
            rect_coords = (
                json_data.get("데이터셋 정보", {})
                .get("데이터셋 상세설명", {})
                .get("렉트좌표", {})
            )
            labeling_info = (
                json_data.get("데이터셋 정보", {})
                .get("데이터셋 상세설명", {})
                .get("라벨링", {})
            )

            # 각 카테고리별로 처리
            for group_name in ["상의", "하의", "아우터", "원피스"]:
                if group_name in rect_coords and rect_coords[group_name]:
                    for bbox_info in rect_coords[group_name]:
                        if not bbox_info:  # 빈 딕셔너리 스킵
                            continue

                        # 바운딩 박스 좌표 추출
                        x = bbox_info.get("X좌표", 0)
                        y = bbox_info.get("Y좌표", 0)
                        w = bbox_info.get("가로", 0)
                        h = bbox_info.get("세로", 0)

                        if w > 0 and h > 0:  # 유효한 박스만
                            # 라벨링에서 해당 그룹의 카테고리 찾기
                            if (
                                group_name in labeling_info
                                and labeling_info[group_name]
                            ):
                                for item in labeling_info[group_name]:
                                    if isinstance(item, dict) and "카테고리" in item:
                                        category = item["카테고리"]
                                        if category in CATEGORIES:
                                            class_id = CATEGORIES.index(category)

                                            # YOLO 형식으로 변환 (x_center, y_center, width, height) - 정규화
                                            x_center = (x + w / 2) / img_width
                                            y_center = (y + h / 2) / img_height
                                            norm_w = w / img_width
                                            norm_h = h / img_height

                                            bboxes.append(
                                                {
                                                    "class_id": class_id,
                                                    "category": category,
                                                    "group": group_name,
                                                    "bbox_norm": [
                                                        x_center,
                                                        y_center,
                                                        norm_w,
                                                        norm_h,
                                                    ],  # 정규화된 좌표
                                                    "bbox_pixel": [
                                                        x,
                                                        y,
                                                        x + w,
                                                        y + h,
                                                    ],  # 픽셀 좌표 (x1, y1, x2, y2)
                                                    "area": w * h,
                                                }
                                            )
                                        break
        except Exception as e:
            print(f"⚠️ 바운딩 박스 추출 오류: {e}")

        return bboxes

    def load_ground_truth_data(self, image_dir, label_dir):
        """JSON 파일에서 ground truth 데이터 로드 (바운딩 박스 + 라벨)"""
        gt_data = {}
        image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]

        print(f"📁 이미지 폴더에서 {len(image_files)}개 파일 발견")

        valid_count = 0
        for img_file in tqdm(image_files, desc="Ground Truth 데이터 로딩 중"):
            json_file = os.path.join(label_dir, img_file.rsplit(".", 1)[0] + ".json")

            if not os.path.exists(json_file):
                continue

            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 이미지 크기 정보
                img_info = data.get("이미지 정보", {})
                img_width = img_info.get("이미지 너비", 800)
                img_height = img_info.get("이미지 높이", 800)

                # 바운딩 박스 추출
                bboxes = self.extract_bbox_from_json(data, img_width, img_height)

                if bboxes:
                    gt_data[img_file] = {
                        "bboxes": bboxes,
                        "img_width": img_width,
                        "img_height": img_height,
                        "num_objects": len(bboxes),
                    }
                    valid_count += 1

            except Exception as e:
                print(f"⚠️ JSON 파일 로딩 오류 ({img_file}): {e}")
                continue

        print(f"✅ 유효한 Ground Truth 데이터 {valid_count}개 로드 완료")
        return gt_data

    def predict_with_boxes(
        self, image_dir, gt_data, conf_threshold=0.5, iou_threshold=0.45
    ):
        """YOLO로 객체 탐지 + 분류 예측 수행"""
        predictions = {}

        print(f"🔍 {len(gt_data)}개 이미지에 대해 예측 수행 중...")

        for img_file in tqdm(gt_data.keys(), desc="YOLO 예측 중"):
            img_path = os.path.join(image_dir, img_file)

            try:
                # YOLO 예측 수행
                results = self.model(
                    img_path, conf=conf_threshold, iou=iou_threshold, verbose=False
                )

                if len(results) == 0 or len(results[0].boxes) == 0:
                    predictions[img_file] = {
                        "boxes": [],
                        "confidences": [],
                        "classes": [],
                        "num_detections": 0,
                    }
                    continue

                # 예측 결과 추출
                boxes = results[0].boxes
                pred_boxes = boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2 형식
                confidences = boxes.conf.cpu().numpy()
                classes = boxes.cls.cpu().numpy().astype(int)

                predictions[img_file] = {
                    "boxes": pred_boxes.tolist(),
                    "confidences": confidences.tolist(),
                    "classes": classes.tolist(),
                    "num_detections": len(pred_boxes),
                }

            except Exception as e:
                print(f"⚠️ 예측 오류 ({img_file}): {e}")
                predictions[img_file] = {
                    "boxes": [],
                    "confidences": [],
                    "classes": [],
                    "num_detections": 0,
                }
                continue

        return predictions

    def calculate_map_metrics(
        self, gt_data, predictions, iou_thresholds=[0.5, 0.75], conf_threshold=0.5
    ):
        """실제 mAP 및 관련 메트릭 계산 (AP 곡선 기반)"""
        print("📊 실제 mAP 메트릭 계산 중...")

        # 모든 예측을 confidence 순으로 정렬하여 수집
        all_detections = []  # [(confidence, class_id, bbox, img_file, pred_idx)]

        # 클래스별 GT 개수 계산
        class_gt_counts = {i: 0 for i in range(NUM_CLASSES)}

        for img_file in gt_data.keys():
            if img_file not in predictions:
                continue

            gt_info = gt_data[img_file]
            pred_info = predictions[img_file]

            # GT 개수 카운트
            for bbox_info in gt_info["bboxes"]:
                class_gt_counts[bbox_info["class_id"]] += 1

            # 예측 박스들을 confidence 순으로 처리
            for i, (pred_bbox, pred_conf, pred_class) in enumerate(
                zip(pred_info["boxes"], pred_info["confidences"], pred_info["classes"])
            ):
                if pred_conf < conf_threshold:
                    continue

                all_detections.append(
                    {
                        "confidence": pred_conf,
                        "class_id": pred_class,
                        "bbox": pred_bbox,
                        "img_file": img_file,
                        "pred_idx": i,
                    }
                )

        # Confidence 순으로 정렬
        all_detections.sort(key=lambda x: x["confidence"], reverse=True)

        map_metrics = {}

        for iou_thresh in iou_thresholds:
            print(f"  IoU Threshold {iou_thresh} 처리 중...")

            # 클래스별 AP 계산
            class_aps = {}
            class_metrics = {}
            total_tp, total_fp, total_fn = 0, 0, 0

            for class_id in range(NUM_CLASSES):
                category_name = CATEGORIES[class_id]

                # 해당 클래스의 예측들만 필터링
                class_detections = [
                    d for d in all_detections if d["class_id"] == class_id
                ]

                if len(class_detections) == 0:
                    # 예측이 없는 경우
                    class_aps[category_name] = 0.0
                    class_metrics[category_name] = {
                        "precision": 0.0,
                        "recall": 0.0,
                        "ap": 0.0,
                        "tp": 0,
                        "fp": 0,
                        "fn": class_gt_counts[class_id],
                    }
                    total_fn += class_gt_counts[class_id]
                    continue

                # AP 계산을 위한 TP/FP 판정
                tp_list = []
                fp_list = []
                matched_gt = set()  # 이미 매칭된 GT 박스들

                for detection in class_detections:
                    img_file = detection["img_file"]
                    pred_bbox = detection["bbox"]

                    if img_file not in gt_data:
                        fp_list.append(1)
                        tp_list.append(0)
                        continue

                    gt_info = gt_data[img_file]

                    # 해당 클래스의 GT 박스들
                    gt_boxes = []
                    gt_indices = []
                    for idx, bbox_info in enumerate(gt_info["bboxes"]):
                        if bbox_info["class_id"] == class_id:
                            gt_boxes.append(bbox_info["bbox_pixel"])
                            gt_indices.append(f"{img_file}_{idx}")

                    if len(gt_boxes) == 0:
                        # 해당 클래스의 GT가 없는 경우
                        fp_list.append(1)
                        tp_list.append(0)
                        continue

                    # IoU 계산하여 최적 매칭 찾기
                    ious = self._calculate_iou_matrix(
                        np.array([pred_bbox]), np.array(gt_boxes)
                    )[
                        0
                    ]  # 첫 번째 행만 사용

                    best_iou_idx = np.argmax(ious)
                    best_iou = ious[best_iou_idx]
                    best_gt_key = gt_indices[best_iou_idx]

                    if best_iou >= iou_thresh and best_gt_key not in matched_gt:
                        # True Positive
                        tp_list.append(1)
                        fp_list.append(0)
                        matched_gt.add(best_gt_key)
                    else:
                        # False Positive
                        tp_list.append(0)
                        fp_list.append(1)

                # AP 계산 (Precision-Recall 곡선의 면적)
                tp_array = np.array(tp_list)
                fp_array = np.array(fp_list)

                # 누적합 계산
                tp_cumsum = np.cumsum(tp_array)
                fp_cumsum = np.cumsum(fp_array)

                # Precision과 Recall 계산
                precision = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
                recall = tp_cumsum / (class_gt_counts[class_id] + 1e-6)

                # AP 계산 (11-point interpolation 방식)
                ap = self._calculate_ap_11point(precision, recall)

                # 최종 메트릭
                final_tp = int(tp_cumsum[-1]) if len(tp_cumsum) > 0 else 0
                final_fp = int(fp_cumsum[-1]) if len(fp_cumsum) > 0 else 0
                final_fn = class_gt_counts[class_id] - final_tp

                final_precision = (
                    final_tp / (final_tp + final_fp)
                    if (final_tp + final_fp) > 0
                    else 0.0
                )
                final_recall = (
                    final_tp / (final_tp + final_fn)
                    if (final_tp + final_fn) > 0
                    else 0.0
                )

                class_aps[category_name] = ap
                class_metrics[category_name] = {
                    "precision": final_precision,
                    "recall": final_recall,
                    "ap": ap,
                    "tp": final_tp,
                    "fp": final_fp,
                    "fn": final_fn,
                }

                total_tp += final_tp
                total_fp += final_fp
                total_fn += final_fn

            # 전체 성능 계산
            overall_precision = (
                total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
            )
            overall_recall = (
                total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
            )
            overall_f1 = (
                2
                * overall_precision
                * overall_recall
                / (overall_precision + overall_recall)
                if (overall_precision + overall_recall) > 0
                else 0
            )

            # 평균 AP (mAP) 계산
            mean_ap = np.mean(list(class_aps.values()))

            map_metrics[f"iou_{iou_thresh}"] = {
                "precision": overall_precision,
                "recall": overall_recall,
                "f1_score": overall_f1,
                "mean_ap": mean_ap,  # 실제 mAP 값
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "class_metrics": class_metrics,
                "class_aps": class_aps,  # 클래스별 AP 값들
            }

        # 전체 mAP 값들
        map_50 = map_metrics.get("iou_0.5", {}).get("mean_ap", 0)
        map_75 = map_metrics.get("iou_0.75", {}).get("mean_ap", 0)
        map_50_95 = np.mean([map_50, map_75])  # 근사값

        return {
            "map_metrics": map_metrics,
            "map_50": map_50,
            "map_75": map_75,
            "map_50_95": map_50_95,
            "total_gt_objects": sum(class_gt_counts.values()),
            "total_pred_objects": len(all_detections),
            "total_images": len(gt_data),
        }

    def _calculate_ap_11point(self, precision, recall):
        """11-point interpolation 방식으로 AP 계산"""
        # 11개 recall 지점에서의 precision 값들
        recall_points = np.linspace(0, 1, 11)
        interpolated_precisions = []

        for r in recall_points:
            # 현재 recall 지점보다 큰 recall들 중에서 최대 precision 찾기
            precisions_at_recall = precision[recall >= r]
            if len(precisions_at_recall) == 0:
                interpolated_precisions.append(0)
            else:
                interpolated_precisions.append(np.max(precisions_at_recall))

        # 평균 계산
        ap = np.mean(interpolated_precisions)
        return ap

    def _calculate_iou_matrix(self, boxes1, boxes2):
        """두 박스 집합 간의 IoU 행렬 계산"""
        if len(boxes1) == 0 or len(boxes2) == 0:
            return np.zeros((len(boxes1), len(boxes2)))

        # 교집합 영역 계산
        x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
        y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
        x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
        y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

        # 합집합 영역 계산
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        union = area1[:, None] + area2[None, :] - intersection

        # IoU 계산
        iou = intersection / (union + 1e-6)
        return iou

    def calculate_classification_metrics(self, gt_data, predictions):
        """분류 성능 메트릭 계산 (기존 방식)"""
        # 분류 성능을 위해 가장 높은 confidence의 예측만 사용
        y_true = []
        y_pred = []
        confidences = []

        for img_file in gt_data.keys():
            if img_file not in predictions:
                continue

            gt_info = gt_data[img_file]
            pred_info = predictions[img_file]

            # GT에서 가장 큰 객체의 클래스 (대표 클래스로 사용)
            if not gt_info["bboxes"]:
                continue

            largest_gt = max(gt_info["bboxes"], key=lambda x: x["area"])
            gt_class = largest_gt["class_id"]

            # 예측에서 가장 높은 confidence의 클래스
            if pred_info["num_detections"] == 0:
                continue

            best_idx = np.argmax(pred_info["confidences"])
            pred_class = pred_info["classes"][best_idx]
            confidence = pred_info["confidences"][best_idx]

            y_true.append(gt_class)
            y_pred.append(pred_class)
            confidences.append(confidence)

        if not y_true:
            return None

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        confidences = np.array(confidences)

        # 기본 분류 메트릭들
        accuracy = accuracy_score(y_true, y_pred)
        balanced_acc = balanced_accuracy_score(y_true, y_pred)

        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            y_true, y_pred, average="micro", zero_division=0
        )
        precision_weighted, recall_weighted, f1_weighted, _ = (
            precision_recall_fscore_support(
                y_true, y_pred, average="weighted", zero_division=0
            )
        )

        # 클래스별 메트릭
        precision_per_class, recall_per_class, f1_per_class, support_per_class = (
            precision_recall_fscore_support(
                y_true, y_pred, average=None, zero_division=0
            )
        )

        # 클래스 분포
        gt_class_counts = Counter(y_true)
        pred_class_counts = Counter(y_pred)

        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

        return {
            "classification_metrics": {
                "accuracy": float(accuracy),
                "balanced_accuracy": float(balanced_acc),
                "macro_precision": float(precision_macro),
                "macro_recall": float(recall_macro),
                "macro_f1": float(f1_macro),
                "micro_precision": float(precision_micro),
                "micro_recall": float(recall_micro),
                "micro_f1": float(f1_micro),
                "weighted_precision": float(precision_weighted),
                "weighted_recall": float(recall_weighted),
                "weighted_f1": float(f1_weighted),
            },
            "confidence_stats": {
                "mean_confidence": float(np.mean(confidences)),
                "std_confidence": float(np.std(confidences)),
                "min_confidence": float(np.min(confidences)),
                "max_confidence": float(np.max(confidences)),
                "median_confidence": float(np.median(confidences)),
            },
            "class_wise_metrics": {
                CATEGORIES[i]: {
                    "precision": (
                        float(precision_per_class[i])
                        if i < len(precision_per_class)
                        else 0.0
                    ),
                    "recall": (
                        float(recall_per_class[i]) if i < len(recall_per_class) else 0.0
                    ),
                    "f1_score": (
                        float(f1_per_class[i]) if i < len(f1_per_class) else 0.0
                    ),
                    "support": (
                        int(support_per_class[i]) if i < len(support_per_class) else 0
                    ),
                    "gt_count": int(gt_class_counts.get(i, 0)),
                    "pred_count": int(pred_class_counts.get(i, 0)),
                }
                for i in range(NUM_CLASSES)
            },
            "class_distribution": {
                "ground_truth": {
                    CATEGORIES[i]: int(gt_class_counts.get(i, 0))
                    for i in range(NUM_CLASSES)
                },
                "predictions": {
                    CATEGORIES[i]: int(pred_class_counts.get(i, 0))
                    for i in range(NUM_CLASSES)
                },
            },
            "confusion_matrix": cm.tolist(),
            "total_samples": len(y_true),
        }

    def save_comprehensive_results(
        self, map_results, classification_results, predictions, gt_data, output_dir
    ):
        """종합 결과 저장 (개선된 CSV 포함, AP 값 추가)"""
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 전체 메트릭 통합
        comprehensive_metrics = {
            "overview": {
                "model_path": self.model_path,
                "total_images": len(gt_data),
                "total_classes": NUM_CLASSES,
                "class_names": CATEGORIES.copy(),
                "evaluation_timestamp": timestamp,
            },
            "detection_metrics": map_results,
            "classification_metrics": classification_results,
        }

        # 1. 종합 메트릭 JSON 저장
        metrics_path = os.path.join(
            output_dir, f"comprehensive_yolo_metrics_{timestamp}.json"
        )
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(comprehensive_metrics, f, ensure_ascii=False, indent=2)
        print(f"📄 종합 메트릭 저장: {metrics_path}")

        # 2. 상세 예측 결과 CSV (기존 유지)
        detailed_results = []
        for img_file in gt_data.keys():
            if img_file not in predictions:
                continue

            gt_info = gt_data[img_file]
            pred_info = predictions[img_file]

            # 각 GT 객체에 대해
            for gt_bbox in gt_info["bboxes"]:
                row = {
                    "image_file": img_file,
                    "gt_category": gt_bbox["category"],
                    "gt_class_id": gt_bbox["class_id"],
                    "gt_group": gt_bbox["group"],
                    "gt_bbox_x1": gt_bbox["bbox_pixel"][0],
                    "gt_bbox_y1": gt_bbox["bbox_pixel"][1],
                    "gt_bbox_x2": gt_bbox["bbox_pixel"][2],
                    "gt_bbox_y2": gt_bbox["bbox_pixel"][3],
                    "gt_area": gt_bbox["area"],
                    "num_predictions": pred_info["num_detections"],
                }

                # 가장 가까운/겹치는 예측 찾기
                if pred_info["num_detections"] > 0:
                    best_match_idx = 0
                    if pred_info["confidences"]:
                        best_match_idx = np.argmax(pred_info["confidences"])

                    row.update(
                        {
                            "pred_category": (
                                CATEGORIES[pred_info["classes"][best_match_idx]]
                                if pred_info["classes"][best_match_idx] < NUM_CLASSES
                                else "unknown"
                            ),
                            "pred_class_id": pred_info["classes"][best_match_idx],
                            "pred_confidence": pred_info["confidences"][best_match_idx],
                            "pred_bbox_x1": pred_info["boxes"][best_match_idx][0],
                            "pred_bbox_y1": pred_info["boxes"][best_match_idx][1],
                            "pred_bbox_x2": pred_info["boxes"][best_match_idx][2],
                            "pred_bbox_y2": pred_info["boxes"][best_match_idx][3],
                            "classification_correct": gt_bbox["class_id"]
                            == pred_info["classes"][best_match_idx],
                        }
                    )
                else:
                    row.update(
                        {
                            "pred_category": "no_detection",
                            "pred_class_id": -1,
                            "pred_confidence": 0.0,
                            "pred_bbox_x1": 0,
                            "pred_bbox_y1": 0,
                            "pred_bbox_x2": 0,
                            "pred_bbox_y2": 0,
                            "classification_correct": False,
                        }
                    )

                detailed_results.append(row)

        if detailed_results:
            detailed_df = pd.DataFrame(detailed_results)
            detailed_csv_path = os.path.join(
                output_dir, f"detailed_detection_results_{timestamp}.csv"
            )
            detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
            print(f"📊 상세 탐지 결과 저장: {detailed_csv_path}")

        # 3. 개선된 카테고리별 성능 요약 CSV (모든 21개 카테고리 포함, AP 값 추가)
        if map_results and "map_metrics" in map_results:
            category_summary = []

            # 모든 21개 카테고리에 대해 처리
            for category in CATEGORIES:
                category_data = {
                    "category": category,
                    "category_id": CATEGORIES.index(category),
                    "group": None,
                }

                # 그룹 정보 추가
                for group_name, categories in CATEGORY_GROUPS.items():
                    if category in categories:
                        category_data["group"] = group_name
                        break

                # IoU별 성능 추가 (AP 값 포함)
                for iou_key, iou_data in map_results["map_metrics"].items():
                    iou_threshold = iou_key.replace("iou_", "")
                    class_metrics = iou_data.get("class_metrics", {})

                    if category in class_metrics:
                        metrics = class_metrics[category]
                    else:
                        metrics = {
                            "precision": 0.0,
                            "recall": 0.0,
                            "ap": 0.0,
                            "tp": 0,
                            "fp": 0,
                            "fn": 0,
                        }

                    # F1-Score 계산
                    f1_score = (
                        2
                        * metrics["precision"]
                        * metrics["recall"]
                        / (metrics["precision"] + metrics["recall"])
                        if (metrics["precision"] + metrics["recall"]) > 0
                        else 0
                    )

                    # IoU별 메트릭 추가 (AP 값 포함)
                    category_data[f"ap_iou_{iou_threshold}"] = round(
                        metrics["ap"], 4
                    )  # **AP 값 추가**
                    category_data[f"precision_iou_{iou_threshold}"] = round(
                        metrics["precision"], 4
                    )
                    category_data[f"recall_iou_{iou_threshold}"] = round(
                        metrics["recall"], 4
                    )
                    category_data[f"f1_score_iou_{iou_threshold}"] = round(f1_score, 4)
                    category_data[f"tp_iou_{iou_threshold}"] = metrics["tp"]
                    category_data[f"fp_iou_{iou_threshold}"] = metrics["fp"]
                    category_data[f"fn_iou_{iou_threshold}"] = metrics["fn"]

                # 분류 성능 추가 (있는 경우)
                if (
                    classification_results
                    and "class_wise_metrics" in classification_results
                ):
                    class_wise = classification_results["class_wise_metrics"].get(
                        category, {}
                    )
                    category_data["classification_precision"] = round(
                        class_wise.get("precision", 0.0), 4
                    )
                    category_data["classification_recall"] = round(
                        class_wise.get("recall", 0.0), 4
                    )
                    category_data["classification_f1"] = round(
                        class_wise.get("f1_score", 0.0), 4
                    )
                    category_data["classification_support"] = class_wise.get(
                        "support", 0
                    )
                    category_data["gt_count"] = class_wise.get("gt_count", 0)
                    category_data["pred_count"] = class_wise.get("pred_count", 0)

                category_summary.append(category_data)

            # CSV 저장
            if category_summary:
                summary_df = pd.DataFrame(category_summary)
                summary_csv_path = os.path.join(
                    output_dir, f"category_performance_summary_{timestamp}.csv"
                )
                summary_df.to_csv(summary_csv_path, index=False, encoding="utf-8-sig")
                print(f"📈 카테고리별 성능 요약 저장 (AP 포함): {summary_csv_path}")

        # 4. 그룹별 성능 요약 CSV (AP 평균 포함)
        if map_results and "map_metrics" in map_results:
            group_summary = []

            for group_name, categories in CATEGORY_GROUPS.items():
                group_data = {
                    "group": group_name,
                    "num_categories": len(categories),
                    "categories": ", ".join(categories),
                }

                # IoU별 그룹 평균 성능 계산 (AP 포함)
                for iou_key, iou_data in map_results["map_metrics"].items():
                    iou_threshold = iou_key.replace("iou_", "")
                    class_metrics = iou_data.get("class_metrics", {})

                    group_aps = []
                    group_precisions = []
                    group_recalls = []
                    group_f1s = []
                    group_tps = []
                    group_fps = []
                    group_fns = []

                    for category in categories:
                        if category in class_metrics:
                            metrics = class_metrics[category]
                        else:
                            metrics = {
                                "precision": 0.0,
                                "recall": 0.0,
                                "ap": 0.0,
                                "tp": 0,
                                "fp": 0,
                                "fn": 0,
                            }

                        f1_score = (
                            2
                            * metrics["precision"]
                            * metrics["recall"]
                            / (metrics["precision"] + metrics["recall"])
                            if (metrics["precision"] + metrics["recall"]) > 0
                            else 0
                        )

                        group_aps.append(metrics["ap"])  # **AP 값 추가**
                        group_precisions.append(metrics["precision"])
                        group_recalls.append(metrics["recall"])
                        group_f1s.append(f1_score)
                        group_tps.append(metrics["tp"])
                        group_fps.append(metrics["fp"])
                        group_fns.append(metrics["fn"])

                    # 그룹 평균 계산 (AP 포함)
                    group_data[f"avg_ap_iou_{iou_threshold}"] = round(
                        np.mean(group_aps), 4
                    )  # **평균 AP 추가**
                    group_data[f"avg_precision_iou_{iou_threshold}"] = round(
                        np.mean(group_precisions), 4
                    )
                    group_data[f"avg_recall_iou_{iou_threshold}"] = round(
                        np.mean(group_recalls), 4
                    )
                    group_data[f"avg_f1_score_iou_{iou_threshold}"] = round(
                        np.mean(group_f1s), 4
                    )
                    group_data[f"total_tp_iou_{iou_threshold}"] = sum(group_tps)
                    group_data[f"total_fp_iou_{iou_threshold}"] = sum(group_fps)
                    group_data[f"total_fn_iou_{iou_threshold}"] = sum(group_fns)

                group_summary.append(group_data)

            # 그룹별 요약 CSV 저장
            if group_summary:
                group_df = pd.DataFrame(group_summary)
                group_csv_path = os.path.join(
                    output_dir, f"group_performance_summary_{timestamp}.csv"
                )
                group_df.to_csv(group_csv_path, index=False, encoding="utf-8-sig")
                print(f"📊 그룹별 성능 요약 저장 (AP 포함): {group_csv_path}")

        return {
            "comprehensive_metrics_json": metrics_path,
            "detailed_results_csv": detailed_csv_path if detailed_results else None,
            "category_performance_csv": (
                summary_csv_path if "summary_csv_path" in locals() else None
            ),
            "group_performance_csv": (
                group_csv_path if "group_csv_path" in locals() else None
            ),
        }

    def print_comprehensive_report(self, map_results, classification_results):
        """종합 성능 리포트 출력 (실제 mAP 포함)"""
        print("\n" + "=" * 100)
        print("🎯 YOLO 종합 성능 검증 리포트 (객체 탐지 + 분류)")
        print("=" * 100)

        # 객체 탐지 성능 (실제 mAP)
        if map_results:
            print(f"\n🎯 객체 탐지 성능 (실제 mAP):")
            print(f"  - mAP@0.5: {map_results.get('map_50', 0):.4f}")
            print(f"  - mAP@0.75: {map_results.get('map_75', 0):.4f}")
            print(f"  - mAP@0.5:0.95: {map_results.get('map_50_95', 0):.4f}")
            print(f"  - 총 GT 객체 수: {map_results.get('total_gt_objects', 0):,}")
            print(f"  - 총 예측 객체 수: {map_results.get('total_pred_objects', 0):,}")

            # IoU@0.5에서의 전체 성능
            if "map_metrics" in map_results and "iou_0.5" in map_results["map_metrics"]:
                iou_50_metrics = map_results["map_metrics"]["iou_0.5"]
                print(f"\n📊 탐지 성능 @IoU=0.5:")
                print(f"  - Precision: {iou_50_metrics.get('precision', 0):.4f}")
                print(f"  - Recall: {iou_50_metrics.get('recall', 0):.4f}")
                print(f"  - F1-Score: {iou_50_metrics.get('f1_score', 0):.4f}")
                print(f"  - Mean AP: {iou_50_metrics.get('mean_ap', 0):.4f}")
                print(
                    f"  - TP: {iou_50_metrics.get('tp', 0)}, FP: {iou_50_metrics.get('fp', 0)}, FN: {iou_50_metrics.get('fn', 0)}"
                )

            # 카테고리별 실제 AP 값 출력 (IoU@0.5) - 모든 21개 카테고리 표시
            if "map_metrics" in map_results and "iou_0.5" in map_results["map_metrics"]:
                iou_50_metrics = map_results["map_metrics"]["iou_0.5"]
                class_metrics = iou_50_metrics.get("class_metrics", {})

                print(f"\n📊 카테고리별 실제 mAP 성능 @IoU=0.5:")
                print(
                    f"{'카테고리':<15} {'AP':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}"
                )
                print("-" * 85)

                # 카테고리 그룹별로 정리 (모든 카테고리 포함)
                for group_name, categories in CATEGORY_GROUPS.items():
                    print(f"\n📂 {group_name}:")
                    for category in categories:
                        if category in class_metrics:
                            metrics = class_metrics[category]
                        else:
                            # 데이터가 없는 카테고리도 표시
                            metrics = {
                                "precision": 0.0,
                                "recall": 0.0,
                                "ap": 0.0,
                                "tp": 0,
                                "fp": 0,
                                "fn": 0,
                            }

                        ap_value = metrics.get("ap", 0.0)
                        f1_score = (
                            2
                            * metrics["precision"]
                            * metrics["recall"]
                            / (metrics["precision"] + metrics["recall"])
                            if (metrics["precision"] + metrics["recall"]) > 0
                            else 0
                        )

                        print(
                            f"  {category:<13} {ap_value:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                            f"{f1_score:<10.4f} {metrics['tp']:<5} {metrics['fp']:<5} {metrics['fn']:<5}"
                        )

            # 카테고리별 실제 AP 값 출력 (IoU@0.75)
            if (
                "map_metrics" in map_results
                and "iou_0.75" in map_results["map_metrics"]
            ):
                iou_75_metrics = map_results["map_metrics"]["iou_0.75"]
                class_metrics = iou_75_metrics.get("class_metrics", {})

                print(f"\n🎯 카테고리별 실제 mAP 성능 @IoU=0.75:")
                print(
                    f"{'카테고리':<15} {'AP':<8} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'TP':<5} {'FP':<5} {'FN':<5}"
                )
                print("-" * 85)

                # 모든 카테고리를 순서대로 표시
                for category in CATEGORIES:
                    if category in class_metrics:
                        metrics = class_metrics[category]
                    else:
                        metrics = {
                            "precision": 0.0,
                            "recall": 0.0,
                            "ap": 0.0,
                            "tp": 0,
                            "fp": 0,
                            "fn": 0,
                        }

                    ap_value = metrics.get("ap", 0.0)
                    f1_score = (
                        2
                        * metrics["precision"]
                        * metrics["recall"]
                        / (metrics["precision"] + metrics["recall"])
                        if (metrics["precision"] + metrics["recall"]) > 0
                        else 0
                    )

                    print(
                        f"  {category:<13} {ap_value:<8.4f} {metrics['precision']:<10.4f} {metrics['recall']:<10.4f} "
                        f"{f1_score:<10.4f} {metrics['tp']:<5} {metrics['fp']:<5} {metrics['fn']:<5}"
                    )

        # 분류 성능 (기존 코드 유지)
        if (
            classification_results
            and "classification_metrics" in classification_results
        ):
            cls_metrics = classification_results["classification_metrics"]
            print(f"\n🏷️ 분류 성능:")
            print(f"  - 정확도: {cls_metrics.get('accuracy', 0):.4f}")
            print(f"  - 균형 정확도: {cls_metrics.get('balanced_accuracy', 0):.4f}")
            print(f"  - Macro F1: {cls_metrics.get('macro_f1', 0):.4f}")
            print(f"  - Weighted F1: {cls_metrics.get('weighted_f1', 0):.4f}")

            # 신뢰도 통계
            if "confidence_stats" in classification_results:
                conf_stats = classification_results["confidence_stats"]
                print(f"\n🔍 예측 신뢰도:")
                print(f"  - 평균: {conf_stats.get('mean_confidence', 0):.4f}")
                print(f"  - 표준편차: {conf_stats.get('std_confidence', 0):.4f}")
                print(
                    f"  - 범위: [{conf_stats.get('min_confidence', 0):.4f}, {conf_stats.get('max_confidence', 0):.4f}]"
                )

        # 최고/최저 성능 클래스 (분류 기준)
        if classification_results and "class_wise_metrics" in classification_results:
            class_metrics = classification_results["class_wise_metrics"]
            class_f1_scores = {
                k: v["f1_score"] for k, v in class_metrics.items() if v["support"] > 0
            }

            if class_f1_scores:
                best_class = max(class_f1_scores, key=class_f1_scores.get)
                worst_class = min(class_f1_scores, key=class_f1_scores.get)

                print(f"\n🏆 클래스별 분류 성능:")
                print(
                    f"  - 최고 성능: {best_class} (F1: {class_f1_scores[best_class]:.4f})"
                )
                print(
                    f"  - 최저 성능: {worst_class} (F1: {class_f1_scores[worst_class]:.4f})"
                )

        # mAP에서 최고/최저 성능 카테고리 (AP 기준)
        if (
            map_results
            and "map_metrics" in map_results
            and "iou_0.5" in map_results["map_metrics"]
        ):
            iou_50_metrics = map_results["map_metrics"]["iou_0.5"]
            if "class_aps" in iou_50_metrics:
                class_aps = iou_50_metrics["class_aps"]

                if class_aps:
                    best_detection_class = max(class_aps, key=class_aps.get)
                    worst_detection_class = min(class_aps, key=class_aps.get)

                    print(f"\n🎯 카테고리별 탐지 성능 (AP@0.5):")
                    print(
                        f"  - 최고 성능: {best_detection_class} (AP: {class_aps[best_detection_class]:.4f})"
                    )
                    print(
                        f"  - 최저 성능: {worst_detection_class} (AP: {class_aps[worst_detection_class]:.4f})"
                    )

        print("=" * 100)

    def print_detailed_category_analysis(self, map_results):
        """카테고리별 상세 분석 출력 (AP 기반)"""
        if not map_results or "map_metrics" not in map_results:
            return

        print(f"\n📊 카테고리별 상세 mAP 분석:")

        for iou_key, iou_data in map_results["map_metrics"].items():
            iou_threshold = iou_key.replace("iou_", "")
            print(f"\n🎯 IoU Threshold: {iou_threshold}")

            if "class_aps" not in iou_data:
                continue

            # 그룹별 평균 AP 계산
            group_performances = {}
            for group_name, categories in CATEGORY_GROUPS.items():
                group_aps = []
                group_precisions = []
                group_recalls = []

                for category in categories:
                    if category in iou_data.get("class_aps", {}):
                        ap = iou_data["class_aps"][category]
                        metrics = iou_data.get("class_metrics", {}).get(category, {})
                        precision = metrics.get("precision", 0.0)
                        recall = metrics.get("recall", 0.0)
                    else:
                        ap = 0.0
                        precision = 0.0
                        recall = 0.0

                    group_aps.append(ap)
                    group_precisions.append(precision)
                    group_recalls.append(recall)

                if group_aps:
                    group_performances[group_name] = {
                        "avg_ap": np.mean(group_aps),
                        "avg_precision": np.mean(group_precisions),
                        "avg_recall": np.mean(group_recalls),
                        "num_categories": len(group_aps),
                    }

            # 그룹별 성능 출력 (AP 기준으로 정렬)
            if group_performances:
                print(f"\n📂 그룹별 평균 성능:")
                for group_name, perf in sorted(
                    group_performances.items(),
                    key=lambda x: x[1]["avg_ap"],
                    reverse=True,
                ):
                    print(
                        f"  {group_name:<10} - AP: {perf['avg_ap']:.4f}, "
                        f"Precision: {perf['avg_precision']:.4f}, "
                        f"Recall: {perf['avg_recall']:.4f} "
                        f"({perf['num_categories']}개 카테고리)"
                    )


def main():
    """메인 실행 함수"""
    print("🚀 YOLO 종합 성능 검증 시작 (객체 탐지 + 분류, 실제 mAP 포함)")

    # 디렉토리 존재 확인
    for path, name in [
        (IMAGE_DIR, "이미지"),
        (LABEL_DIR, "라벨"),
        (MODEL_PATH, "모델"),
    ]:
        if not os.path.exists(path):
            print(f"❌ {name} 경로가 존재하지 않습니다: {path}")
            return

    try:
        # 검증기 초기화
        validator = YOLODetectionValidator(MODEL_PATH)

        # Ground Truth 데이터 로드
        print(f"\n📂 Ground Truth 데이터 로딩...")
        gt_data = validator.load_ground_truth_data(IMAGE_DIR, LABEL_DIR)

        if len(gt_data) == 0:
            print("❌ 유효한 Ground Truth 데이터가 없습니다.")
            return

        # YOLO 예측 수행
        print(f"\n🔮 YOLO 예측 수행...")
        predictions = validator.predict_with_boxes(IMAGE_DIR, gt_data)

        # 실제 mAP 메트릭 계산
        print(f"\n📊 실제 mAP 메트릭 계산...")
        map_results = validator.calculate_map_metrics(gt_data, predictions)

        # 분류 메트릭 계산
        print(f"\n🏷️ 분류 메트릭 계산...")
        classification_results = validator.calculate_classification_metrics(
            gt_data, predictions
        )

        # 결과 저장
        print(f"\n💾 결과 저장...")
        saved_files = validator.save_comprehensive_results(
            map_results, classification_results, predictions, gt_data, RESULTS_DIR
        )

        # 종합 리포트 출력 (카테고리별 실제 mAP 포함)
        validator.print_comprehensive_report(map_results, classification_results)

        # 상세 카테고리 분석 출력 (AP 기반)
        validator.print_detailed_category_analysis(map_results)

        print(f"\n✅ 종합 검증 완료!")
        print(f"📁 결과 파일들이 저장된 위치: {RESULTS_DIR}")
        for file_type, file_path in saved_files.items():
            if file_path:
                print(f"  - {file_type}: {os.path.basename(file_path)}")

    except Exception as e:
        print(f"❌ 검증 중 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
