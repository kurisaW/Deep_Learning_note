# coding=utf-8
'''
@ Summary: 将图片转存为 k210 的模型输入格式。CHW
@ Update:  

@ file:    save_chw_img.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/5/20 16:41

@ Example:
           python save_chw_img.py ../Datasets/quantize_data/1.png ../Applications/test_data/new_img2_chw.h 28x28

'''
import sys
import cv2
from pathlib import Path


def save_chw_img_data(image_path, dataset_h, shape=(28, 28)):
    """just for k210"""
    image_raw = cv2.imread(image_path)
    image = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, shape)
    # cv2.imshow("input", image)
    # cv2.waitKey(1000)
    # cv2.destroyWindow("gray")

    dataset_h = Path(dataset_h)

    with open(dataset_h, "w+") as f:
        print(f"#ifndef _{dataset_h.stem.upper()}_H_\n#define _{dataset_h.stem.upper()}_H_\n", file=f)
        print(f"// {image_path} {shape}, HW", file=f)
        print(f"const static uint8_t {dataset_h.stem.upper()}[] __attribute__((aligned(128))) = {'{'}", file=f)
        # print(", ".join([str(i) for i in image.flatten()]), file=f)
        print(", ".join(map(lambda i: str(i), image.flatten())), file=f)
        print("};\n\n#endif", file=f)

    print(f"save image to {dataset_h} done.")
    return image_raw


def main():
    image_path = sys.argv[1]
    dataset_h = sys.argv[2]
    # (width, height)
    resize_shape = sys.argv[3].split('x')
    resize_shape = (int(resize_shape[0]), int(resize_shape[1]))

    image = save_chw_img_data(image_path, dataset_h, resize_shape)

    # # save resize image
    # image_path = Path(image_path)
    # new_img_path = image_path.parent / f"{image_path.stem}_320_240.jpg"
    # image = cv2.resize(image, (320, 240))
    # cv2.imwrite(str(new_img_path), image)


if __name__ == "__main__":
    main()