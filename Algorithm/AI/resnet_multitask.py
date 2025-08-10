import os
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============ 설정값 ============
try:
    from backend.config.config import MODEL_PATHS, CLASS_MAPPINGS
except ImportError:
    from config.config import MODEL_PATHS, CLASS_MAPPINGS


# ============ ResNet50 기반 다중 작업 모델 ============
class MultiTaskResNet50(nn.Module):
    def __init__(self, mappings):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        feat_dim = 2048
        # category 제외 속성별 head
        self.heads = nn.ModuleDict(
            {
                k: nn.Linear(feat_dim, len(v))
                for k, v in mappings.items()
                if k != "category"
            }
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {k: head(feat) for k, head in self.heads.items()}


# ============ 속성 예측기 클래스 ============
class FashionAttributePredictor:
    def __init__(self, device: str = None, top_k: int = 3):
        """
        MODEL_PATHS['resnet'] 의 .pth 파일을 불러와 ResNet50 multitask를 초기화
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = top_k
        # 모델 생성 및 가중치 로드
        resnet_path = MODEL_PATHS.get("resnet")
        if resnet_path is None or not os.path.exists(resnet_path):
            raise FileNotFoundError(
                f"ResNet 체크포인트를 찾을 수 없습니다: {resnet_path}"
            )
        self.model = MultiTaskResNet50(CLASS_MAPPINGS).to(self.device)
        state = torch.load(resnet_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.eval()
        # 이미지 전처리
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_attributes(self, crop, class_mappings=None):
        """
        crop: numpy array (HWC, BGR or RGB) or PIL.Image.Image
        class_mappings: dict (default: CLASS_MAPPINGS)
        각 속성별 top_k 예측 결과를 반환
        """
        from PIL import Image as PILImage
        import numpy as np

        mappings = class_mappings if class_mappings is not None else CLASS_MAPPINGS
        # numpy array면 PIL.Image로 변환
        if isinstance(crop, np.ndarray):
            # BGR to RGB 변환 필요시
            if crop.shape[2] == 3:
                crop = PILImage.fromarray(crop[..., ::-1])
            else:
                crop = PILImage.fromarray(crop)
        elif not isinstance(crop, PILImage.Image):
            raise ValueError("crop은 numpy array 또는 PIL.Image여야 합니다.")
        x = self.transform(crop).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        results = {}
        for attr, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(
                probs, min(self.top_k, probs.size(1)), dim=1
            )
            results[attr] = [
                {
                    "class_name": mappings[attr][idx.item()],
                    "probability": float(top_probs[0, i].item()),
                }
                for i, idx in enumerate(top_idxs[0])
            ]
        return results

    def predict_image(self, image_path: str) -> dict:
        """
        주어진 이미지에 대해 각 속성별 top_k 예측 결과를 반환
        """
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outputs = self.model(x)
        results = {}
        for attr, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(
                probs, min(self.top_k, probs.size(1)), dim=1
            )
            results[attr] = [
                {
                    "class_name": CLASS_MAPPINGS[attr][idx.item()],
                    "probability": float(top_probs[0, i].item()),
                }
                for i, idx in enumerate(top_idxs[0])
            ]
        return results
