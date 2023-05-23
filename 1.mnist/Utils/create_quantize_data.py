# coding=utf-8
'''
@ Summary: 从 Test 数据集中随机抽取 200 张图片，在 NNcase 量化模型时候会用到
@ Update:

@ file:    creat_quantize_data.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/8/23 14:49
'''
import cv2
import random
import numpy as np
from pathlib import Path
from tensorflow.keras import datasets


def create_q_data(root_img_path, q_data_path, keep_save=200):
    assert root_img_path.exists(), "No {root_img_path} found..."

    q_data_path.mkdir(exist_ok=True)

    if list(q_data_path.rglob("*.png")):
        print("No need to creating quantization dataset!")
        return

    # prepare test data
    (_, _), (x_test, _) = datasets.mnist.load_data(root_img_path.resolve()/'mnist.npz')
    all_imgs = random.sample(x_test.tolist(), keep_save)
    for i, img in enumerate(all_imgs):
        cv2.imwrite(str(q_data_path/f"{i+1}.png"), np.array(img))

    print("Save quantization images done...")


def main():
    root_img_path = Path("../Datasets/train_data")
    q_data_path = Path("../Datasets/quantize_data")
    create_q_data(root_img_path, q_data_path)


if __name__ == "__main__":
    main()