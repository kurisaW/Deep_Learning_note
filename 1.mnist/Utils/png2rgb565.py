# coding=utf-8
'''
@ Summary: img2rgb565() 函数将图片保存为 RGB565
           show_rgb565() 将rgb565反转为rgb888并显示图片
           输入是 240x320 的图片
@ Update:  

@ file:    png2rgb565.py
@ version: 1.0.0

@ Author:  Lebhoryi@gmail.com
@ Date:    2021/5/11 18:41
@ Link:    https://github.com/jimmywong2003/PNG-to-RGB565/blob/master/png2rgb565.py

@ Example:
           python .\png2rgb565.py ..\k210-mnist-project\images\3.jpg ..\k210-mnist-project\applications\test_data\img3.h

@ Update:  1. 新增图片是否显示；
           2. 新增数据头部定义
           3. 将图片从 pillow 读取改为 cv2 读取
           4. 变量命名改为根据 .h 文件名来命名
@ Date:    2021/06/07

@ Update:  增加变量对齐
@ Date:    2021/06/08
'''
import sys
import cv2
import numpy as np
from pathlib import Path

isSWAP = False

def img2rgb565(save_hex=True):
    len_argument = len(sys.argv)
    if (len_argument != 3):
        print("")
        print("Correct Usage:")
        print("\tpython png2rgb565.py <png_file> <include_file>")
        print("")
        sys.exit(0)


    try:
        im = cv2.imread(sys.argv[1])
        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        # image resize
        im = cv2.resize(im, (28, 28))
    except:
        raise Exception(f"Fail to open png file {sys.argv[1]}")

    image_height = im.shape[0]
    image_width = im.shape[1]

    try:
        outfile = open(sys.argv[2], "w")
    except:
        raise Exception(f"Can't write the file {sys.argv[2]}")
    
    file_name = Path(sys.argv[2])
        
    print(f"#ifndef _{file_name.stem.upper()}_H_\n#define _{file_name.stem.upper()}_H_\n", file=outfile)
    print(f"// {sys.argv[1]} Width:{image_width} Height:{image_height}", file=outfile)
    print(f"const static uint16_t {file_name.stem.upper()}[] __attribute__((aligned(128))) = {'{'}", file=outfile)

    pix = im  # load pixel array
    for h in range(image_height):
        for w in range(image_width):
            if ((h * 16 + w) % 16 == 0):
                print(" ", file=outfile)
                print("\t\t", file=outfile, end='')

            if w < image_width:
                R = pix[h][w][0] >> 3
                G = pix[h][w][1] >> 2
                B = pix[h][w][2] >> 3

                rgb = (R << 11) | (G << 5) | B

                if (isSWAP == True):
                    swap_string_low = rgb >> 8
                    swap_string_high = (rgb & 0x00FF) << 8
                    swap_string = swap_string_low | swap_string_high
                    print("0x%04x, " % (swap_string), file=outfile, end='')
                else:
                    if save_hex:
                        print("0x%04x, " % (rgb), file=outfile, end='')
                    else:
                        print("%04d, " % (rgb), file=outfile, end='')
            else:
                rgb = 0

    print("", file=outfile)
    print("};\n", file=outfile)
    print("#endif", file=outfile)

    outfile.close()

    print(f"Image file {sys.argv[1]} converted to {sys.argv[2]} done.")
    return sys.argv[2], (image_height, image_width)


def show_rgb565(img_file, shape=(224, 224)):
    # Read 16-bit RGB565 image into array of uint16
    with open(img_file, 'r') as f:
        lines = f.read().split()
    try:
        lines = lines[15:-2]
        lines = list(map(lambda i: int(i[:-1]), lines))
        image = np.array(lines)
        rgb565array = np.reshape(image, shape)
    except:
        raise Exception("wrong split list...")

    # Pick up image dimensions
    h, w = rgb565array.shape

    # Make a numpy array of matching shape, but allowing for 8-bit/channel for R, G and B
    rgb888array = np.zeros([h, w, 3], dtype=np.uint8)

    for row in range(h):
        for col in range(w):
            # Pick up rgb565 value and split into rgb888
            rgb565 = rgb565array[row, col]
            r = ((rgb565 >> 11) & 0x1f) << 3
            g = ((rgb565 >> 5) & 0x3f) << 2
            b = ((rgb565) & 0x1f) << 3
            # Populate result array
            rgb888array[row, col] = r, g, b

    # Save result as PNG
    # Image.fromarray(rgb888array).save('}result.png')
    rgb888array = cv2.cvtColor(rgb888array, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", rgb888array)
    cv2.waitKey(2000)
    cv2.destroyWindow("result")


def main():
    save_hex = True
    img_file, shape = img2rgb565(save_hex)
    if not save_hex:
        show_rgb565(img_file, shape)

if __name__ == "__main__":
    main()