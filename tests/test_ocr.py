# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: test_ocr.py
# author: RPA高老师
# description: OCR模块测试

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr import OCRRunner
from src.config import get_settings


def test_ocr_locate_keyword():
    """测试OCR关键词定位功能"""
    settings = get_settings()
    settings.ensure_dirs()
    
    # 初始化OCR
    ocr = OCRRunner(
        use_gpu=settings.ocr.use_gpu,
        lang=settings.ocr.lang,
        det_limit_side_len=settings.ocr.det_limit_side_len,
    )
    
    # 测试图片路径（需要放置测试图片）
    test_image = os.path.join(settings.paths.input_dir, "2025-12-27_122321.png")
    
    if not os.path.exists(test_image):
        print(f"[跳过] 测试图片不存在: {test_image}")
        print("请将测试图片放置到 data/input/test_input.png")
        return
    
    # 执行关键词定位
    results = ocr.locate_keyword(
        image_path=test_image,
        keyword="动态",
        draw=True,
        output_path=os.path.join(settings.paths.output_dir, "test_ocr_result.png"),
    )
    
    print(f"[测试] 找到 {len(results)} 个关键词")
    for i, r in enumerate(results, 1):
        print(f"  #{i}: {r}")
    
    print("[通过] OCR关键词定位测试完成")


if __name__ == "__main__":
    test_ocr_locate_keyword()
