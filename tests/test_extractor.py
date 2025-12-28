# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: test_extractor.py
# author: RPA高老师
# description: 封面提取器测试

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.ocr import OCRRunner
from src.extractor import CoverExtractorCV
from src.config import get_settings


def test_cover_extraction():
    """测试封面提取功能"""
    settings = get_settings()
    settings.ensure_dirs()
    
    # 测试图片路径
    test_image = os.path.join(settings.paths.input_dir, "2025-12-27_122321.png")
    
    if not os.path.exists(test_image):
        print(f"[跳过] 测试图片不存在: {test_image}")
        print("请将测试图片放置到 data/input/test_input.png")
        return
    
    # 1. 初始化OCR定位关键词
    ocr = OCRRunner(
        use_gpu=settings.ocr.use_gpu,
        lang=settings.ocr.lang,
        det_limit_side_len=settings.ocr.det_limit_side_len,
    )
    
    results = ocr.locate_keyword(
        image_path=test_image,
        keyword="动态",
        draw=False,
    )
    
    if not results:
        print("[失败] 未找到'动态'关键词")
        return
    
    keyword_box = results[0]['box']
    print(f"[测试] 找到关键词，box坐标: {keyword_box}")
    
    # 2. 初始化封面提取器
    extractor = CoverExtractorCV(
        bg_color_hex=settings.extractor.bg_color_hex,
        color_tolerance=settings.extractor.color_tolerance,
        min_cover_area=settings.extractor.min_cover_area,
        aspect_ratio_range=settings.extractor.aspect_ratio_range,
        edge_crop=settings.extractor.edge_crop,
    )
    
    # 3. 提取封面
    extracted = extractor.extract_covers(
        image_path=test_image,
        keyword_box=keyword_box,
        output_dir=settings.paths.covers_dir,
        debug=True,
    )
    
    print(f"[测试] 共提取 {len(extracted)} 个封面")
    print("[通过] 封面提取测试完成")


def test_cover_extraction_without_keyword():
    """测试不使用关键词的封面提取"""
    settings = get_settings()
    settings.ensure_dirs()
    
    test_image = os.path.join(settings.paths.input_dir, "test_input.png")
    
    if not os.path.exists(test_image):
        print(f"[跳过] 测试图片不存在: {test_image}")
        return
    
    extractor = CoverExtractorCV(
        bg_color_hex=settings.extractor.bg_color_hex,
        color_tolerance=settings.extractor.color_tolerance,
    )
    
    extracted = extractor.extract_covers(
        image_path=test_image,
        keyword_box=None,  # 不使用关键词
        output_dir=os.path.join(settings.paths.output_dir, "covers_full"),
        debug=True,
    )
    
    print(f"[测试] 从整图提取 {len(extracted)} 个封面")
    print("[通过] 无关键词封面提取测试完成")


if __name__ == "__main__":
    print("=" * 50)
    print("封面提取器测试")
    print("=" * 50)
    
    test_cover_extraction()
    print()
    # test_cover_extraction_without_keyword()
