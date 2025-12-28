# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: ocr_engine.py
# author: RPA高老师

from paddleocr import PaddleOCR

# ⚠️ 只初始化一次，避免反复加载模型
ocr = PaddleOCR(
    use_gpu=True,
    use_angle_cls=True,
    lang="ch",
    show_log=False,
    det_limit_side_len=2048  # 防止超大图直接进模型
)
