import torch
import torch.nn as nn


class YOLOv11MultiTask:
    def __init__(self, yolo_model):
        self.yolo = yolo_model

    def detect(self, image, conf_thres=0.5, iou_thres=0.45):
        """
        객체 탐지(카테고리)만 수행. 결과는 YOLO의 predict 결과(boxes, conf, cls 등) 반환.
        """
        return self.yolo.predict(
            source=image, conf=conf_thres, iou=iou_thres, verbose=False
        )[0]

    def extract_crops(self, image, results, conf_thres=0.5):
        """
        탐지된 객체의 crop(잘린 이미지), bbox, conf, cls만 반환. 세부 속성 분류는 하지 않음.
        """
        crops = []
        if hasattr(results, "boxes"):
            for box, conf, cls in zip(
                results.boxes.xyxy.cpu().numpy(),
                results.boxes.conf.cpu().numpy(),
                results.boxes.cls.cpu().numpy(),
            ):
                if conf < conf_thres:
                    continue
                x1, y1, x2, y2 = map(int, box)
                crop = image[y1:y2, x1:x2]
                crops.append((crop, (x1, y1, x2, y2), float(conf), int(cls)))
        return crops
