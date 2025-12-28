# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: run.py
# author: RPA高老师
# description: 项目主入口 - 封面提取工具

import os
import sys
import argparse

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.ocr import OCRRunner
from src.extractor import CoverExtractorCV
from src.config import get_settings


def extract_covers(
    image_path: str,
    keyword: str = "动态",
    output_dir: str = None,
    debug: bool = False,
):
    """
    从长截图中提取封面
    
    Args:
        image_path: 输入图片路径
        keyword: 定位关键词
        output_dir: 输出目录
        debug: 是否开启调试模式
    """
    settings = get_settings()
    settings.ensure_dirs()
    
    if output_dir is None:
        output_dir = settings.paths.covers_dir
    
    print("=" * 50)
    print("微信视频号封面提取工具")
    print("=" * 50)
    
    # 1. 初始化OCR
    print("\n[1/3] 初始化OCR识别器...")
    ocr = OCRRunner(
        use_gpu=settings.ocr.use_gpu,
        lang=settings.ocr.lang,
        det_limit_side_len=settings.ocr.det_limit_side_len,
    )
    
    # 2. 定位关键词
    print(f"\n[2/3] 定位关键词 '{keyword}'...")
    results = ocr.locate_keyword(
        image_path=image_path,
        keyword=keyword,
        draw=debug,
        output_path=os.path.join(settings.paths.debug_dir, "keyword_result.png") if debug else None,
    )
    
    if not results:
        print(f"[错误] 未找到关键词 '{keyword}'")
        return []
    
    keyword_box = results[0]['box']
    print(f"  找到 {len(results)} 个匹配")
    print(f"  使用第一个匹配的坐标: {keyword_box}")
    
    # 3. 提取封面
    print(f"\n[3/3] 提取封面图片...")
    extractor = CoverExtractorCV(
        bg_color_hex=settings.extractor.bg_color_hex,
        color_tolerance=settings.extractor.color_tolerance,
        min_cover_area=settings.extractor.min_cover_area,
        aspect_ratio_range=settings.extractor.aspect_ratio_range,
        edge_crop=settings.extractor.edge_crop,
    )
    
    extracted = extractor.extract_covers(
        image_path=image_path,
        keyword_box=keyword_box,
        output_dir=output_dir,
        debug=debug,
    )
    
    print("\n" + "=" * 50)
    print(f"提取完成！共 {len(extracted)} 个封面")
    print(f"输出目录: {output_dir}")
    print("=" * 50)
    
    return extracted


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(
        description="微信视频号封面提取工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="输入图片路径",
    )
    parser.add_argument(
        "-k", "--keyword",
        default="动态",
        help="定位关键词（默认: 动态）",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出目录（默认: data/output/covers）",
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="开启调试模式，生成调试图片",
    )
    
    args = parser.parse_args()
    
    # 如果没有提供图片路径，使用默认测试图片
    if args.image is None:
        settings = get_settings()
        args.image = os.path.join(settings.paths.input_dir, "input.png")
        print(f"[提示] 未指定图片路径，使用默认路径: {args.image}")
    
    if not os.path.exists(args.image):
        print(f"[错误] 图片不存在: {args.image}")
        return
    
    extract_covers(
        image_path=args.image,
        keyword=args.keyword,
        output_dir=args.output,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
