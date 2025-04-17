import cv2
import numpy as np
import os
from glob import glob

def compute_yellow_value(image_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 전체 바나나 마스크
    lower_banana = np.array([15, 20, 20])
    upper_banana = np.array([40, 255, 255])
    banana_mask = cv2.inRange(img_hsv, lower_banana, upper_banana)

    # 노란색 마스크
    lower_yellow = np.array([20, 100, 150])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    yellow_on_banana = cv2.bitwise_and(yellow_mask, banana_mask)

    banana_pixels = cv2.countNonZero(banana_mask)
    yellow_pixels = cv2.countNonZero(yellow_on_banana)

    yellow = yellow_pixels
    return yellow

def compute_banana_freshness(image_path, yellow):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 바나나 전체 마스크
    lower_banana = np.array([15, 20, 20])
    upper_banana = np.array([40, 255, 255])
    banana_mask = cv2.inRange(img_hsv, lower_banana, upper_banana)

    # 노란색 마스크
    lower_yellow = np.array([20, 100, 200])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    yellow_on_banana = cv2.bitwise_and(yellow_mask, banana_mask)

    banana_pixels = cv2.countNonZero(banana_mask)
    yellow_pixels = cv2.countNonZero(yellow_on_banana)

    freshness_ratio = yellow_pixels / (yellow*1.05)

    return round(freshness_ratio, 8)

def process_and_save_labels(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob(os.path.join(input_folder, "*.png")))

    # 보정값 계산 (1.png 기준)
    yellow_image_path = os.path.join(input_folder, "001.png")
    yellow = compute_yellow_value(yellow_image_path)
    print(f"보정값(correction): {yellow}")

    for path in image_files:
        filename = os.path.basename(path)
        num = os.path.splitext(filename)[0]  # 예: "1.png" → "1"
        ratio = compute_banana_freshness(path, yellow)

        txt_path = os.path.join(output_folder, f"{num}.txt")
        with open(txt_path, "w") as f:
            f.write(f"{str(int(num))}\n{ratio:.8f}")

    print(f"완료: {len(image_files)}개의 txt 파일이 생성되었습니다.")

# 실행
input_folder = "Dataset/banana_2/image"
output_folder = "Dataset/banana_2/label"
process_and_save_labels(input_folder, output_folder)
