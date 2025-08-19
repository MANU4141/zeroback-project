import numpy as np
from PIL import Image


class YOLOv11MultiTask:
    def __init__(self, yolo_model):
        self.yolo = yolo_model

    def detect(self, image, conf_thres, iou_thres):
        """
        객체 탐지(카테고리)만 수행. 결과는 YOLO의 predict 결과(boxes, conf, cls 등) 반환.
        """
        try:
            results = self.yolo.predict(
                source=image, conf=conf_thres, iou=iou_thres, verbose=False
            )
            return results[0] if results and len(results) > 0 else None
        except Exception:
            # YOLO 추론 실패 시 None 반환
            return None

    def extract_crops(self, image, results, conf_thres):
        """
        탐지된 객체의 crop(잘린 이미지), bbox, conf, cls만 반환. 세부 속성 분류는 하지 않음.
        박스 좌표 클램핑, 검증, 방어 처리 포함.

        좌표계 가정: YOLO 결과는 입력 이미지 좌표계 기준 (전처리 리사이즈/패딩은 외부에서 처리됨)
        """
        crops = []

        # 결과/입력 방어 처리
        if results is None or not hasattr(results, "boxes") or results.boxes is None:
            return crops

        # 이미지 형식 통일 및 크기 확보
        if isinstance(image, Image.Image):
            image = np.array(image)

        if not isinstance(image, np.ndarray) or image.ndim < 2:
            return crops

        height, width = image.shape[:2]

        # 박스가 비어있는 경우 처리
        if len(results.boxes) == 0:
            return crops

        try:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            classes = results.boxes.cls.cpu().numpy()
        except Exception:
            # GPU/텐서 처리 실패 시 안전하게 빈 리스트 반환
            return crops

        for box, conf, cls in zip(boxes, confs, classes):
            # conf 재필터 안전장치
            if conf < conf_thres:
                continue

            # 박스 좌표 클램핑 및 검증 (round 후 클램핑으로 1px 손실 방지)
            x1, y1, x2, y2 = box
            x1 = max(0, min(int(round(x1)), width - 1))
            y1 = max(0, min(int(round(y1)), height - 1))
            x2 = max(0, min(int(round(x2)), width))
            y2 = max(0, min(int(round(y2)), height))

            # 유효하지 않은 박스 스킵
            if x2 <= x1 or y2 <= y1:
                continue

            # crop 추출 및 크기 검증
            try:
                crop = image[y1:y2, x1:x2]
                if crop.size == 0:
                    continue
                crops.append((crop, (x1, y1, x2, y2), float(conf), int(cls)))
            except Exception:
                # crop 추출 실패 시 해당 박스 스킵
                continue

        return crops
