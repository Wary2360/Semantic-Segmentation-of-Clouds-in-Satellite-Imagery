import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from SatelliteCloudGenerator.src import *
from tqdm import tqdm

# PALETTE 정의
PALETTE = {
    (0, 0, 0): 0,  # background
    (255, 0, 0): 1,  # thick_cloud
    (0, 255, 0): 2,  # thin_cloud
    (255, 255, 0): 3  # cloud_shadow
}

# 이미지와 마스크 파일 경로를 가져옵니다.
ngr_dir = './dataset/224_194/ngr'
label_dir = './dataset/224_194/label'
output_dir = './dataset/224_194/cloud'

os.makedirs(output_dir, exist_ok=True)

ngr_files = [os.path.join(ngr_dir, f) for f in os.listdir(ngr_dir) if f.endswith('.png')]
label_files = [os.path.join(label_dir, f) for f in os.listdir(label_dir) if f.endswith('.png')]

# 레이블이 0과 3만 있는 이미지를 필터링합니다.
filtered_ngr_files = []
filtered_label_files = []

for ngr_file, label_file in tqdm(zip(ngr_files, label_files), desc='Filtering', total=len(ngr_files)):
    label_img = cv2.imread(label_file)
    label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2RGB)
    label = np.zeros((label_img.shape[0], label_img.shape[1]), dtype=np.uint8)
    
    for color, value in PALETTE.items():
        mask = np.all(label_img == color, axis=-1)
        label[mask] = value
    
    unique_labels = np.unique(label)
    if set(unique_labels).issubset({0, 3}):
        filtered_ngr_files.append(ngr_file)
        filtered_label_files.append(label_file)

# 필터링된 이미지에 thick_cloud를 추가합니다.
for ngr_file, label_file in tqdm(zip(filtered_ngr_files, filtered_label_files), desc='Adding cloud', total=len(filtered_ngr_files)):
    ngr_img = cv2.imread(ngr_file)[...,:3] / 255
    label_img = cv2.imread(label_file)
    ngr_img = torch.FloatTensor(ngr_img).permute(2, 0, 1).unsqueeze(0)

    # thick_cloud 추가
    cl, mask = add_cloud(ngr_img,
                     min_lvl=0.0,
                     max_lvl=0.3,
                     cloud_color=False,
                     channel_offset=0,
                     blur_scaling=2.0,
                     return_cloud=True
                    )

    cl_path = os.path.join(output_dir, os.path.basename(ngr_file).replace('.png', '_cloud.png'))
    mask_path = os.path.join(output_dir, os.path.basename(ngr_file).replace('.png', '_mask.png'))

    cl = cl.squeeze().permute(1, 2, 0).numpy() * 255.0
    sim_mask = mask.squeeze().permute(1, 2, 0).numpy()
    label_img[sim_mask[:,:,0] >= 0.15] = [0, 255, 0]
    
    
    cv2.imwrite(cl_path, cl)
    cv2.imwrite(mask_path, label_img)
