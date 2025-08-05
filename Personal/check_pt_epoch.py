import torch

# last.pt 파일 경로
pt_path = R"C:\Users\rkdgu\Desktop\VSC_clone_zeroback\zeroback-project\Algorithm\AI\models\last (10).pt"

# 모델 가중치 로드
ckpt = torch.load(pt_path, map_location="cpu")

print(
    ckpt.keys()
)  # YOLOv5는 대부분 'model', 'optimizer', 'epoch', 'best_fitness' 등 존재

# 에폭 정보가 있으면 표시
if "epoch" in ckpt:
    print(f"마지막 학습 에폭: {ckpt['epoch']}")
else:
    print("이 weight 파일에는 에폭 정보가 없습니다.")
