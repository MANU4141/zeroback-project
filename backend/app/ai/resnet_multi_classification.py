import os
import json
import logging
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


# ============ 설정값 ============
try:
    from config import Config
except Exception:
    # Fallback for direct script execution
    from config import Config  # type: ignore


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
        resnet_path = Config.MODEL_PATHS.get("resnet")
        if resnet_path is None or not os.path.exists(resnet_path):
            raise FileNotFoundError(
                f"ResNet 체크포인트를 찾을 수 없습니다: {resnet_path}"
            )
        self.model = MultiTaskResNet50(Config.CLASS_MAPPINGS).to(self.device)
        state = torch.load(resnet_path, map_location=self.device)
        # 일부 체크포인트 키/차원 불일치에 유연하게 대응
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model_state = self.model.state_dict()
        filtered_state = {}
        mismatched = []
        for k, v in state.items():
            if k in model_state:
                if hasattr(v, "shape") and hasattr(model_state[k], "shape"):
                    if tuple(v.shape) == tuple(model_state[k].shape):
                        filtered_state[k] = v
                    else:
                        mismatched.append(
                            (k, tuple(v.shape), tuple(model_state[k].shape))
                        )
                else:
                    # 텐서가 아닌 경우는 건너뜀
                    mismatched.append((k, "<non-tensor>", "<tensor>"))
            # 모델에 없는 키는 자동으로 무시됨 (strict=False)

        if mismatched:
            logging.getLogger(__name__).warning(
                "ResNet 체크포인트에서 차원 불일치 키를 무시합니다: %s",
                ", ".join([f"{k} ({src}->{dst})" for k, src, dst in mismatched]),
            )

        # 체크포인트 로딩 상태 점검
        if not filtered_state:
            logging.getLogger(__name__).warning(
                "체크포인트에서 로드된 파라미터가 없습니다. 모든 레이어가 랜덤 초기화 상태일 수 있습니다."
            )

        missing_after = [k for k in model_state.keys() if k not in filtered_state]
        if missing_after:
            total_params = len(model_state)
            missing_ratio = len(missing_after) / total_params
            if missing_ratio > 0.5:  # 50% 이상 누락 시 강한 경고
                logging.getLogger(__name__).warning(
                    "랜덤 초기화 파라미터가 전체의 %.1f%%입니다. 모델 성능에 영향을 줄 수 있습니다.",
                    missing_ratio * 100,
                )
            else:
                logging.getLogger(__name__).info(
                    "랜덤 초기화 유지 파라미터(일부): %s%s",
                    ", ".join(missing_after[:10]),
                    "..." if len(missing_after) > 10 else "",
                )

        self.model.load_state_dict(filtered_state, strict=False)
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

        mappings = (
            class_mappings if class_mappings is not None else Config.CLASS_MAPPINGS
        )
        # numpy array면 PIL.Image로 변환
        if isinstance(crop, np.ndarray):
            # 그레이스케일(2D) → RGB(3D)
            if crop.ndim == 2:
                crop = np.stack([crop] * 3, axis=-1)
            # RGBA → RGB (알파 채널 제거)
            elif crop.ndim == 3 and crop.shape[2] == 4:
                crop = crop[..., :3]
            # 유효하지 않은 차원 체크
            if crop.ndim != 3 or crop.shape[2] != 3:
                raise ValueError("이미지 배열은 HxWx3 형식이어야 합니다.")
            # BGR → RGB 변환 (OpenCV 기본 포맷 가정)
            crop = PILImage.fromarray(crop[..., ::-1])
        elif not isinstance(crop, PILImage.Image):
            raise ValueError("crop은 numpy array 또는 PIL.Image여야 합니다.")
        x = self.transform(crop).unsqueeze(0).to(self.device)
        from contextlib import nullcontext

        with torch.no_grad():
            if self.device.startswith("cuda"):
                ctx = torch.amp.autocast("cuda")
            else:
                ctx = nullcontext()
            with ctx:
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
        from contextlib import nullcontext

        with torch.no_grad():
            if self.device.startswith("cuda"):
                ctx = torch.amp.autocast("cuda")
            else:
                ctx = nullcontext()
            with ctx:
                outputs = self.model(x)
        results = {}
        for attr, logits in outputs.items():
            probs = torch.softmax(logits, dim=1)
            top_probs, top_idxs = torch.topk(
                probs, min(self.top_k, probs.size(1)), dim=1
            )
            results[attr] = [
                {
                    "class_name": Config.CLASS_MAPPINGS[attr][idx.item()],
                    "probability": float(top_probs[0, i].item()),
                }
                for i, idx in enumerate(top_idxs[0])
            ]
        return results
