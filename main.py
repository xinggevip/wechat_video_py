# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: main.py
# author: RPA高老师
from ocr.ocr_runner import OCRRunner


def main():
    # 1. 初始化 OCR（生产建议：程序启动时只初始化一次）
    ocr = OCRRunner(
        use_gpu=True,            # 没有 CUDA 就改成 False
        lang="ch",
        det_limit_side_len=960,  # 长截图建议 960~1280
    )

    # 2. 调用关键词定位
    results = ocr.locate_keyword(
        image_path=r"src/input.png",   # 长截图路径
        keyword="动态",          # 要查找的关键词
        draw=True,               # 是否画红框
        output_path="result.png" # 输出图片
    )

    # 3. 打印结果
    print(f"命中数量：{len(results)}")
    for i, r in enumerate(results, 1):
        print(f"#{i}", r)


if __name__ == "__main__":
    main()
