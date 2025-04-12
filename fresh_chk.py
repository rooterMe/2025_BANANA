import cv2
import numpy as np
import os
from glob import glob

def segment_and_colorize(image_path, output_path):
    img = cv2.imread(image_path)
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 그림자를 제외한 바나나 전체 마스크 (V값을 높게 설정해 그림자 제거)
    lower_banana = np.array([10, 20, 20])
    upper_banana = np.array([40, 255, 255])
    banana_mask = cv2.inRange(img_hsv, lower_banana, upper_banana)

    # 노란색 범위 (upper 값 낮춤: 너무 밝거나 선명한 노랑 제외)
    lower_yellow = np.array([18, 100, 100])
    upper_yellow = np.array([35, 255, 255])  # upper 낮춤

    yellow_mask = cv2.inRange(img_hsv, lower_yellow, upper_yellow)

    # 갈변된 바나나 = 바나나 마스크 - 노란색 마스크
    brown_mask = cv2.bitwise_and(banana_mask, cv2.bitwise_not(yellow_mask))

    # 결과 이미지: 기본 배경은 초록색
    result_img = np.full_like(img, (0, 255, 0))  # 초록색

    # 노란 부분 → 빨간색
    result_img[yellow_mask > 0] = (0, 0, 255)

    # 갈변된 부분 → 파란색
    result_img[brown_mask > 0] = (255, 0, 0)

    cv2.imwrite(output_path, result_img)

def process_and_save_labels(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    image_files = sorted(glob(os.path.join(input_folder, "*.png")))

    for path in image_files:
        filename = os.path.basename(path)
        output_path = os.path.join(output_folder, filename)
        segment_and_colorize(path, output_path)

    print(f"{len(image_files)}개의 이미지가 {output_folder}에 저장되었습니다.")

# 실행
input_folder = "Dataset/banana_0_seq"
output_folder = "Dataset/banana_0_test"
process_and_save_labels(input_folder, output_folder)
