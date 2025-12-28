# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: main.py
# author: RPA高老师

from ocr_runner import ocr_long_image

if __name__ == "__main__":
    image_path = "src/input.png"
    image_path = r'C:\Users\Admin\Desktop\2025-12-27_095600.png'
    target_text = "动态"   # 可为 None，表示全部返回
    target_text = "品质"

    output_image_path = "output_marked.png"

    results = ocr_long_image(
        image_path=image_path,
        target_text=target_text,
        output_image_path=output_image_path
    )

    print(f"命中数量：{len(results)}")
    for r in results:
        print(r)

    print(f"已输出画框图片：{output_image_path}")
