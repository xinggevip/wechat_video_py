# -*- coding: utf-8 -*-
# time: 2025/12/27 22:57
# file: draw_boxes.py
# author: RPA高老师

import cv2
from typing import List, Dict


def draw_boxes_on_image(
    image,
    ocr_results: List[Dict],
    color=(0, 0, 255),
    thickness=2
):
    """
    在图片上绘制 OCR 识别框
    """
    output = image.copy()

    for item in ocr_results:
        box = item["box"]  # 四点坐标

        # 转成 OpenCV 需要的格式
        pts = [(int(p[0]), int(p[1])) for p in box]

        # 画四条线
        for i in range(4):
            cv2.line(
                output,
                pts[i],
                pts[(i + 1) % 4],
                color,
                thickness
            )

    return output
