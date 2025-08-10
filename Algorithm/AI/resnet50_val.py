import os
import sys
import json
import torch
import torch.nn as nn
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.metrics import f1_score
import pandas as pd

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# 설정 불러오기
from config.config import MODEL_PATHS, CLASS_MAPPINGS

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = r"D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\images"
LABEL_DIR = r"D:\end_github_zeroback\zeroback-project\Algorithm\DATASET\labels"
BATCH_SIZE = 16
NUM_WORKERS = 0  # 디버깅 시 워커 없이 실행

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


class MultiTaskResNet50(nn.Module):
    def __init__(self, mappings):
        super().__init__()
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.heads = nn.ModuleDict(
            {k: nn.Linear(2048, len(v)) for k, v in mappings.items() if k != "category"}
        )

    def forward(self, x):
        feat = self.backbone(x)
        return {k: head(feat) for k, head in self.heads.items()}


class FashionAttributePredictor:
    def __init__(self, device=None, top_k=3):
        self.device = device or DEVICE
        self.top_k = top_k
        path = MODEL_PATHS.get("resnet")
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        self.model = MultiTaskResNet50(CLASS_MAPPINGS).to(self.device)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.eval()
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def predict_image(self, image_path):
        img = Image.open(image_path).convert("RGB")
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            outs = self.model(x)
        results = {}
        for attr, logits in outs.items():
            probs = torch.softmax(logits, dim=1)
            top_p, top_i = torch.topk(probs, min(self.top_k, probs.size(1)), dim=1)
            results[attr] = [
                {
                    "class_name": CLASS_MAPPINGS[attr][idx.item()],
                    "probability": float(top_p[0, i].item()),
                }
                for i, idx in enumerate(top_i[0])
            ]
        return results


class FashionValDataset(Dataset):
    def __init__(self, img_dir, lbl_dir, mappings, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.mappings = mappings
        self.transform = transform
        self.files = [f for f in os.listdir(img_dir) if f.lower().endswith(".jpg")]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fn = self.files[idx]
        img = Image.open(os.path.join(self.img_dir, fn)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        labels = {k: -1 for k in self.mappings}
        jp = os.path.join(self.lbl_dir, fn.rsplit(".", 1)[0] + ".json")
        if os.path.exists(jp):
            data = json.load(open(jp, encoding="utf-8"))
            lab = (
                data.get("데이터셋 정보", {})
                .get("데이터셋 상세설명", {})
                .get("라벨링", {})
            )
            for part in lab.values():
                for item in part:
                    if not isinstance(item, dict):
                        continue
                    for kor, v in item.items():
                        eng = KOR_TO_ENG.get(kor)
                        if eng is None or v in (None, "", []):
                            continue
                        vals = v if isinstance(v, list) else [v]
                        for vv in vals:
                            if vv in self.mappings[eng]:
                                labels[eng] = self.mappings[eng].index(vv)
                                break
                        if labels[eng] != -1:
                            continue
        return img, labels


def collate_fn(batch):
    imgs = torch.stack([b[0] for b in batch], dim=0)
    lbls = {
        k: torch.tensor([b[1][k] for b in batch], dtype=torch.long) for k in batch[0][1]
    }
    return imgs, lbls


def validate(model, loader):
    model.eval()
    cat_t = defaultdict(lambda: defaultdict(list))
    cat_p = defaultdict(lambda: defaultdict(list))
    pbar = tqdm(
        loader,
        desc="Validating",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
    )
    with torch.no_grad():
        for imgs, labels in pbar:
            imgs = imgs.to(DEVICE)
            outs = model(imgs)
            cats = labels["category"].tolist()
            for i, c in enumerate(cats):
                if c < 0:
                    continue
                cn = CLASS_MAPPINGS["category"][c]
                for attr, out in outs.items():
                    tl = labels[attr][i].item()
                    if tl < 0:
                        continue
                    pr = out[i].argmax().item()
                    cat_t[cn][attr].append(tl)
                    cat_p[cn][attr].append(pr)
    res = {}
    for cn, attrs in cat_t.items():
        fs = {}
        for attr, tr in attrs.items():
            pr = cat_p[cn][attr]
            fs[attr] = f1_score(tr, pr, average="macro", zero_division=0) if tr else 0.0
        res[cn] = fs
    return res


if __name__ == "__main__":
    # 1) 예측 테스트
    predictor = FashionAttributePredictor(device="cuda", top_k=3)
    test_img = r"D:\github_zeroback\zeroback-project\Algorithm\AI\images\test.jpg"
    print(json.dumps(predictor.predict_image(test_img), ensure_ascii=False, indent=2))

    # 2) 검증 지표
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    ds = FashionValDataset(IMAGE_DIR, LABEL_DIR, CLASS_MAPPINGS, transform)
    _, val_ds = random_split(ds, [int(len(ds) * 0.8), len(ds) - int(len(ds) * 0.8)])
    loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collate_fn,
    )

    stats = validate(predictor.model, loader)
    for cat, metr in stats.items():
        df = pd.DataFrame.from_dict(metr, orient="index", columns=["F1 Score"])
        df.index.name = "Attribute"
        print(f"\n=== Category: {cat} ===")
        print(df.to_markdown())
