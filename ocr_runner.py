# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: ocr_runner.py
# author: RPA高老师
import cv2
from ocr_engine import ocr
from long_image_split import split_long_image
from draw_boxes import draw_boxes_on_image


def ocr_long_image(
    image_path: str,
    target_text: str = None,
    output_image_path: str = None
):
    """
    对长截图进行 OCR
    - 可选 target_text：只保留命中项
    - 可选 output_image_path：输出画框图片
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("图片读取失败")

    slices = split_long_image(image)
    final_results = []

    for y_start, y_end, img_slice in slices:
        result = ocr.ocr(img_slice, cls=True)

        if not result or not result[0]:
            continue

        for line in result[0]:
            box = line[0]
            text = line[1][0]
            score = float(line[1][1])

            if target_text and target_text not in text:
                continue

            # 坐标回算到原图
            mapped_box = [
                [int(p[0]), int(p[1] + y_start)]
                for p in box
            ]

            final_results.append({
                "text": text,
                "confidence": score,
                "box": mapped_box
            })

    # === 新增：画框并保存 ===
    if output_image_path and final_results:
        marked = draw_boxes_on_image(image, final_results)
        cv2.imwrite(output_image_path, marked)

    return final_results
