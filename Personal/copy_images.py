import os
import shutil
from tqdm import tqdm
import time


def copy_images_from_labels(label_folder, image_root_folder, output_folder):
    # 출력 폴더가 없으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 레이블 파일 리스트 가져오기
    label_files = [f for f in os.listdir(label_folder) if f.endswith(".txt")]

    # 레이블명(확장자 제거) 리스트 생성
    label_names = [os.path.splitext(f)[0] for f in label_files]

    # image_root_folder 하위 전체 이미지 파일 탐색 (이름:경로 딕셔너리)
    image_files_dict = {}
    for root, dirs, files in os.walk(image_root_folder):
        for file in files:
            if file.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
                name = os.path.splitext(file)[0]
                image_files_dict[name] = os.path.join(root, file)

    # 파일 복사 과정 및 진행 게이지 출력
    copied_count = 0
    start_time = time.time()
    for label_name in tqdm(label_names, desc="Copying images", ncols=100):
        if label_name in image_files_dict:
            src = image_files_dict[label_name]
            dst = os.path.join(output_folder, os.path.basename(src))
            shutil.copy2(src, dst)
            copied_count += 1

    elapsed_time = time.time() - start_time
    print(f"\n총 복사된 이미지 수: {copied_count}")
    print(f"총 소요 시간: {elapsed_time:.2f}초")


# 사용 예시: 아래 경로를 본인 환경에 맞춰 수정해서 사용하세요.
label_folder = (
    R"C:\Users\rkdgu\Desktop\VSC_clone_zeroback\zeroback-project\Personal\labels"
)
image_root_folder = R"D:\clothes_AI\fashion_recommendation\training\datasets\K-Fashion\Validation\images"
output_folder = (
    R"C:\Users\rkdgu\Desktop\VSC_clone_zeroback\zeroback-project\Personal\images"
)
copy_images_from_labels(label_folder, image_root_folder, output_folder)
