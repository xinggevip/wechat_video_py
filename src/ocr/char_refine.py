# -*- coding: utf-8 -*-
# time: 2025/12/28 09:57
# file: char_refine.py
# author: RPA高老师
# description: 字符级精细定位模块

import cv2
import numpy as np
from paddleocr import PaddleOCR

# 全局单例，防止反复加载模型
_ocr_char = None


def _get_ocr_char():
    """获取字符识别OCR实例（懒加载）"""
    global _ocr_char
    if _ocr_char is None:
        _ocr_char = PaddleOCR(
            use_gpu=True,
            lang="ch",
            det=False,
            rec=True,
            cls=False
        )
    return _ocr_char


def char_level_refine(line_img, offset, keyword, min_char_width=6):
    """
    字符级精细定位
    
    Args:
        line_img: 行图像（已裁剪）
        offset: (x_offset, y_offset) 行在整图中的位置
        keyword: 目标关键词
        min_char_width: 最小字符宽度
        
    Returns:
        精确 box（list），可能为空
    """
    ox, oy = offset
    ocr_char = _get_ocr_char()

    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    proj = np.sum(binary, axis=0)

    chars = []
    in_char = False
    start = 0

    for i, v in enumerate(proj):
        if v > 0 and not in_char:
            in_char = True
            start = i
        elif v == 0 and in_char:
            in_char = False
            if i - start >= min_char_width:
                chars.append((start, i))

    if in_char:
        chars.append((start, len(proj)))

    char_infos = []

    for sx, ex in chars:
        char_img = line_img[:, sx:ex]
        res = ocr_char.ocr(char_img, cls=False)
        if not res or not res[0]:
            continue

        char_infos.append({
            "text": res[0][0][1][0],
            "sx": sx,
            "ex": ex
        })

    full_text = "".join(c["text"] for c in char_infos)
    if keyword not in full_text:
        return []

    idx = full_text.find(keyword)
    start = char_infos[idx]["sx"]
    end = char_infos[idx + len(keyword) - 1]["ex"]

    h = line_img.shape[0]

    return [[
        [ox + start, oy],
        [ox + end,   oy],
        [ox + end,   oy + h],
        [ox + start, oy + h],
    ]]
