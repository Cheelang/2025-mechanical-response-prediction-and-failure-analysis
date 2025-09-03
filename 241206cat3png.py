import os
import cv2
import numpy as np


a_folder = r""
b_folder = r""
c_folder = r""
d_folder = r""


os.makedirs(d_folder, exist_ok=True)


file_names = [f for f in os.listdir(a_folder) if f.endswith('.png')]


for file_name in file_names:

    a_path = os.path.join(a_folder, file_name)
    b_path = os.path.join(b_folder, file_name)
    c_path = os.path.join(c_folder, file_name)
    d_path = os.path.join(d_folder, file_name)


    if not (os.path.exists(b_path) and os.path.exists(c_path)):
        print(f"文件 {file_name} 在 b 或 c 文件夹中不存在，跳过...")
        continue


    a_gray = cv2.imread(a_path, cv2.IMREAD_GRAYSCALE)
    b_gray = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)
    c_gray = cv2.imread(c_path, cv2.IMREAD_GRAYSCALE)


    if a_gray.shape != b_gray.shape or a_gray.shape != c_gray.shape:
        print(f"文件 {file_name} 的图片尺寸不一致，跳过...")
        continue


    merged_img = cv2.merge([a_gray, b_gray, c_gray])


    cv2.imwrite(d_path, merged_img)

print("处理完成！所有图片已保存到文件夹:", d_folder)