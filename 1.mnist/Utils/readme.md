## 文件说明

```shell
D:.
    creat_quantize_data.py  # 生成量化的数据集（随机）

    png2rgb565.py  # 将图片保存成 RGB565 格式
    readme.md
    save_chw_img.py  # 将图片保存成 chw 格式

```

k210 输入的图片格式：CHW，一般图片保存格式：HWC

k210 LCD ILI9341 显示的图片格式：RGB565，一般图片保存格式：RGB888

## RGB888 转 RGB565

> 用这个的原因是需要将 k210-mnist-project/images/logo_320_240.jpg 文件转存为 logo_image.h 文件，
>
> 目的是为了开机显示 RT-Thread Logo

关于 `png2rgb565.py` 使用说明：

```shell
# 使用
python png2rgb565.py logo_320_240.jpg logo_image.h
```

默认是以 "0x%04x" 十六进制保存的。两个函数

- img2rgb565()： 图片保存为 RGB565
- show_rgb565(): 上述逆过程，RGB565转 RGB888，只显示不保存，验证 RGB565是否转存成功
  - 使用该函数需要将125行变量改为 False 即可
