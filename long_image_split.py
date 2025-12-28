# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: long_image_split.py
# author: RPA高老师

import cv2
from typing import List, Tuple


def split_long_image(
    image,
    slice_height: int = 1200,
    overlap: int = 200
) -> List[Tuple[int, int, any]]:
    """
    将长图纵向切分
    返回：(y_start, y_end, image_slice)
    """
    h, w = image.shape[:2]
    slices = []

    y = 0
    while y < h:
        y_end = min(y + slice_height, h)
        crop = image[y:y_end, :]
        slices.append((y, y_end, crop))

        if y_end == h:
            break
        y = y_end - overlap  # 保留重叠区域

    return slices
