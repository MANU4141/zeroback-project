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
from datetime import datetime
import glob

# 프로젝트 루트 추가
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..")))

# 설정 불러오기
from config.config import CLASS_MAPPINGS

# 현재 파일 위치를 기준으로 프로젝트 구조 경로 계산
current_dir = os.path.dirname(__file__)  # AI 폴더
algorithm_dir = os.path.dirname(current_dir)  # Algorithm 폴더
project_root = os.path.dirname(algorithm_dir)  # 프로젝트 루트

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_DIR = os.path.join(project_root, "backend", "DATA", "images")
LABEL_DIR = os.path.join(project_root, "backend", "DATA", "labels")
BATCH_SIZE = 16
NUM_WORKERS = 0

# 모델 폴더 경로 설정
MODELS_DIR = os.path.join(project_root, "AI", "ResNet50_summary", "EXP")
RESULTS_DIR = os.path.join(project_root, "AI", "ResNet50_summary", "RESULTS")

# ✨ 4개 주요 카테고리 정의
SELECTED_CATEGORIES = ["아우터", "상의", "하의", "원피스"]

# ✨ 21개 세부 카테고리 → 4개 대분류 매핑
CATEGORY_MAPPING = {
    # 상의
    "탑": "상의",
    "블라우스": "상의",
    "티셔츠": "상의",
    "니트웨어": "상의",
    "셔츠": "상의",
    "브라탑": "상의",
    "후드티": "상의",
    # 하의
    "청바지": "하의",
    "팬츠": "하의",
    "스커트": "하의",
    "레깅스": "하의",
    "조거팬츠": "하의",
    # 아우터
    "코트": "아우터",
    "재킷": "아우터",
    "점퍼": "아우터",
    "패딩": "아우터",
    "베스트": "아우터",
    "가디건": "아우터",
    "짚업": "아우터",
    # 원피스
    "드레스": "원피스",
    "점프수트": "원피스",
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
    def __init__(self, model_path, device=None, top_k=3):
        self.device = device or DEVICE
        self.top_k = top_k
        self.model_path = model_path

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")

        self.model = MultiTaskResNet50(CLASS_MAPPINGS).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
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


def count_labels_per_category_attribute(
    loader, selected_categories, class_mappings, category_mapping
):
    """각 카테고리(대분류), 속성별 실제 라벨 개수 카운트"""
    label_counts = defaultdict(lambda: defaultdict(int))

    for imgs, labels in tqdm(loader, desc="Counting labels", leave=False):
        cats = labels["category"].tolist()
        batch_size = len(cats)

        for i in range(batch_size):
            c = cats[i]
            if c < 0:
                continue
            cn = class_mappings["category"][c]
            main_cat = category_mapping.get(cn)
            if main_cat is None or main_cat not in selected_categories:
                continue

            for attr in labels:
                if attr == "category":
                    continue
                label_val = labels[attr][i].item()
                if label_val >= 0:  # 유효한 라벨만 카운트
                    label_counts[main_cat][attr] += 1

    return label_counts


def validate(model, loader, model_name, selected_categories=None):
    """단일 모델 검증 (특정 카테고리만 성능계산) + 레이블 개수 포함"""
    if selected_categories is None:
        selected_categories = SELECTED_CATEGORIES

    model.eval()
    cat_t = defaultdict(lambda: defaultdict(list))
    cat_p = defaultdict(lambda: defaultdict(list))

    # ✨ 먼저 레이블 개수 카운트
    print(f"📊 레이블 개수 계산 중...")
    label_counts = count_labels_per_category_attribute(
        loader, selected_categories, CLASS_MAPPINGS, CATEGORY_MAPPING
    )

    pbar = tqdm(
        loader,
        desc=f"Validating {model_name} (4개 카테고리)",
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

                # 세부 카테고리를 대분류로 매핑
                main_category = CATEGORY_MAPPING.get(cn)
                if main_category is None or main_category not in selected_categories:
                    continue

                for attr, out in outs.items():
                    tl = labels[attr][i].item()
                    if tl < 0:
                        continue
                    pr = out[i].argmax().item()
                    cat_t[main_category][attr].append(tl)
                    cat_p[main_category][attr].append(pr)

    # F1 스코어 계산 + 레이블 개수 포함
    results = {}
    for cn, attrs in cat_t.items():
        fs = {}
        for attr, tr in attrs.items():
            pr = cat_p[cn][attr]
            f1 = f1_score(tr, pr, average="macro", zero_division=0) if tr else 0.0
            total_labels = label_counts[cn][attr]  # ✨ 총 레이블 개수
            fs[attr] = {"f1_score": f1, "total_labels": total_labels}
        results[cn] = fs

    return results


def validate_multiple_models(models_dir, results_dir):
    """여러 모델 검증 및 결과 저장 (4개 카테고리만) + 레이블 개수 포함"""

    os.makedirs(results_dir, exist_ok=True)
    model_files = glob.glob(os.path.join(models_dir, "*.pth"))

    if not model_files:
        print(f"No .pth files found in {models_dir}")
        return

    print(f"Found {len(model_files)} model files")
    print(f"검증 대상 카테고리: {', '.join(SELECTED_CATEGORIES)}")

    # 데이터셋 준비
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

    all_results = {}
    detailed_results = []

    # 각 모델에 대해 검증 수행
    for model_file in model_files:
        model_name = os.path.basename(model_file).replace(".pth", "")
        print(f"\n{'='*50}")
        print(f"Validating model: {model_name}")
        print(f"{'='*50}")

        try:
            predictor = FashionAttributePredictor(model_file, device=DEVICE)
            stats = validate(predictor.model, loader, model_name, SELECTED_CATEGORIES)
            all_results[model_name] = stats

            # ✨ 상세 결과 수집 (레이블 개수 포함)
            for category, metrics in stats.items():
                for attribute, metric_data in metrics.items():
                    detailed_results.append(
                        {
                            "model_name": model_name,
                            "category": category,
                            "attribute": attribute,
                            "f1_score": metric_data["f1_score"],
                            "total_labels": metric_data["total_labels"],
                        }
                    )

            # ✨ 개별 모델 결과 출력 (레이블 개수 포함)
            print(f"\nResults for {model_name} (4개 주요 카테고리):")
            for cat, metrics in stats.items():
                print(f"\n--- Category: {cat} ---")
                df_data = {
                    "F1 Score": [metrics[attr]["f1_score"] for attr in metrics],
                    "Total Labels": [metrics[attr]["total_labels"] for attr in metrics],
                }
                df = pd.DataFrame(df_data, index=list(metrics.keys()))
                df.index.name = "Attribute"
                print(df.to_string(float_format="%.4f"))

        except Exception as e:
            print(f"Error validating model {model_name}: {str(e)}")
            continue

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if detailed_results:
        # ✨ 1. 상세 결과를 CSV로 저장 (레이블 개수 포함)
        detailed_df = pd.DataFrame(detailed_results)
        detailed_csv_path = os.path.join(
            results_dir,
            f"detailed_validation_results_4categories_with_counts_{timestamp}.csv",
        )
        detailed_df.to_csv(detailed_csv_path, index=False, encoding="utf-8-sig")
        print(f"\n상세 결과 저장됨: {detailed_csv_path}")

        # ✨ 2. F1 스코어만으로 피벗 테이블 생성
        summary_df = detailed_df.pivot_table(
            index=["category", "attribute"],
            columns="model_name",
            values="f1_score",
            fill_value=0,
        )
        summary_csv_path = os.path.join(
            results_dir,
            f"summary_validation_results_4categories_with_counts_{timestamp}.csv",
        )
        summary_df.to_csv(summary_csv_path, encoding="utf-8-sig")
        print(f"요약 결과 저장됨: {summary_csv_path}")

        # ✨ 3. 레이블 개수 피벗 테이블 별도 저장
        labels_df = detailed_df.pivot_table(
            index=["category", "attribute"],
            columns="model_name",
            values="total_labels",
            fill_value=0,
            aggfunc="first",  # 모든 모델에서 레이블 개수는 동일하므로 첫 번째 값 사용
        )
        labels_csv_path = os.path.join(
            results_dir, f"label_counts_by_category_attribute_{timestamp}.csv"
        )
        labels_df.to_csv(labels_csv_path, encoding="utf-8-sig")
        print(f"레이블 개수 저장됨: {labels_csv_path}")

        # 4. 모델별 평균 성능 계산
        model_avg_df = (
            detailed_df.groupby("model_name")["f1_score"]
            .agg(["mean", "std", "count"])
            .round(4)
        )
        model_avg_df.columns = ["Mean_F1", "Std_F1", "Num_Attributes"]
        model_avg_path = os.path.join(
            results_dir,
            f"model_average_performance_4categories_with_counts_{timestamp}.csv",
        )
        model_avg_df.to_csv(model_avg_path, encoding="utf-8-sig")
        print(f"모델 평균 성능 저장됨: {model_avg_path}")

    # 5. 전체 결과를 JSON으로 저장 (레이블 개수 포함)
    json_path = os.path.join(
        results_dir, f"full_validation_results_4categories_with_counts_{timestamp}.json"
    )
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"전체 결과 JSON 저장됨: {json_path}")

    # 최고 성능 모델 찾기
    if detailed_results:
        best_model = detailed_df.groupby("model_name")["f1_score"].mean().idxmax()
        best_score = detailed_df.groupby("model_name")["f1_score"].mean().max()
        print(f"\n🏆 최고 성능 모델: {best_model} (평균 F1: {best_score:.4f})")

    return all_results


def create_comparison_report(results_dir, results_data):
    """모델 비교 리포트 생성 (4개 카테고리) + 레이블 개수 포함"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 모델별 전체 평균 성능 + 총 레이블 개수
    model_performance = {}
    for model_name, categories in results_data.items():
        all_scores = []
        total_labels = 0
        for cat_metrics in categories.values():
            for metric_data in cat_metrics.values():
                all_scores.append(metric_data["f1_score"])
                total_labels += metric_data["total_labels"]

        if all_scores:
            model_performance[model_name] = {
                "average_f1": sum(all_scores) / len(all_scores),
                "num_metrics": len(all_scores),
                "total_labels": total_labels,
            }

    # 리포트 생성
    report_path = os.path.join(
        results_dir, f"model_comparison_report_4categories_with_counts_{timestamp}.txt"
    )
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("패션 속성 예측 모델 검증 리포트 (4개 주요 카테고리 + 레이블 개수)\n")
        f.write("=" * 70 + "\n")
        f.write(f"생성 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"검증 모델 수: {len(results_data)}\n")
        f.write(f"검증 카테고리: {', '.join(SELECTED_CATEGORIES)}\n\n")

        f.write("모델별 전체 평균 성능:\n")
        f.write("-" * 50 + "\n")

        sorted_models = sorted(
            model_performance.items(), key=lambda x: x[1]["average_f1"], reverse=True
        )

        for i, (model_name, perf) in enumerate(sorted_models, 1):
            f.write(
                f"{i}. {model_name}: {perf['average_f1']:.4f} "
                f"(총 {perf['num_metrics']}개 지표, {perf['total_labels']}개 레이블)\n"
            )

    print(f"비교 리포트 저장됨: {report_path}")


if __name__ == "__main__":
    # 기본값을 동적 경로로 설정
    default_models_dir = os.path.join(project_root, "AI", "ResNet50_summary", "EXP")
    default_results_dir = os.path.join(project_root, "AI", "ResNet50_summary", "RESULTS")
    
    MODELS_DIR = input("모델 폴더 경로를 입력하세요: ").strip()
    if not MODELS_DIR:
        MODELS_DIR = default_models_dir

    RESULTS_DIR = input("결과 저장 폴더 경로를 입력하세요 (엔터시 기본값): ").strip()
    if not RESULTS_DIR:
        RESULTS_DIR = default_results_dir

    print(f"모델 폴더: {MODELS_DIR}")
    print(f"결과 저장: {RESULTS_DIR}")
    print(f"검증 대상: {', '.join(SELECTED_CATEGORIES)}")

    results = validate_multiple_models(MODELS_DIR, RESULTS_DIR)

    if results:
        create_comparison_report(RESULTS_DIR, results)
        print(f"\n✅ 검증 완료! 총 {len(results)}개 모델이 검증되었습니다.")
        print(f"📁 결과 파일들이 {RESULTS_DIR}에 저장되었습니다.")
        print(f"🎯 검증된 카테고리: {', '.join(SELECTED_CATEGORIES)}")
    else:
        print("❌ 검증할 수 있는 모델이 없습니다.")
