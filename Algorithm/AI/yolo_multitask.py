import torch
import torch.nn as nn


class YOLOv11MultiTask(nn.Module):
    def __init__(
        self, yolo_model, num_classes_dict, feature_dim=512, backbone_slice=11
    ):
        super().__init__()
        self.yolo = yolo_model
        self.backbone = self.yolo.model.model[:backbone_slice]
        self.feature_dim = feature_dim
        self.classifiers = nn.ModuleDict(
            {
                task: nn.Linear(self.feature_dim, n_cls)
                for task, n_cls in num_classes_dict.items()
            }
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        feat = self.pool(x)
        feat = self.flatten(feat)
        return {task: head(feat) for task, head in self.classifiers.items()}

    def detect(self, image, conf_thres=0.5, iou_thres=0.45):
        return self.yolo.predict(
            source=image, conf=conf_thres, iou=iou_thres, verbose=False
        )[0]

    def extract_crops(self, image, results, conf_thres=0.5):
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

    def predict_attributes(self, crop, CLASS_MAPPINGS, device="cpu", top_k=3):
        self.classifiers.eval()
        with torch.no_grad():
            import cv2

            img = cv2.resize(crop, (640, 640))
            img = img[..., ::-1].transpose(2, 0, 1) / 255.0
            img = torch.tensor(img, dtype=torch.float32, device=device).unsqueeze(0)
            outputs = self.forward(img)
            result = {}
            for task, out in outputs.items():
                probs = torch.softmax(out, dim=1)
                top_probs, top_idxs = torch.topk(probs, top_k, dim=1)
                result[task] = [
                    {
                        "class_name": CLASS_MAPPINGS[task][top_idxs[0, i].item()],
                        "probability": float(top_probs[0, i].item()),
                    }
                    for i in range(top_k)
                ]
            return result
