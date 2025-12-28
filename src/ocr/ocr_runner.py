# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: ocr_runner.py
# author: RPA高老师
# description: OCR识别器，支持大图分块识别

import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Tuple


class OCRRunner:
    """OCR识别器，支持关键词定位和大图分块识别"""
    
    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "ch",
        det_limit_side_len: int = 960,
    ):
        """
        初始化OCR识别器
        
        Args:
            use_gpu: 是否使用GPU加速
            lang: 识别语言
            det_limit_side_len: 检测限制边长
        """
        self.det_limit_side_len = det_limit_side_len
        self.ocr = PaddleOCR(
            use_gpu=use_gpu,
            lang=lang,
            det=True,
            rec=True,
            cls=False,
            det_limit_side_len=det_limit_side_len,
            show_log=False,
        )

    def _split_image(self, img: np.ndarray, overlap: int = 100) -> List[Tuple[np.ndarray, int, int]]:
        """
        将大图片分割成多个小块进行识别
        
        Args:
            img: 原始图片
            overlap: 分块之间的重叠像素，防止文字被切断
            
        Returns:
            [(分块图片, y偏移量, x偏移量), ...]
        """
        h, w = img.shape[:2]
        block_size = self.det_limit_side_len
        
        # 如果图片尺寸小于等于block_size，不需要分块
        if h <= block_size and w <= block_size:
            return [(img, 0, 0)]
        
        blocks = []
        y = 0
        while y < h:
            x = 0
            block_h = min(block_size, h - y)
            while x < w:
                block_w = min(block_size, w - x)
                block = img[y:y + block_h, x:x + block_w]
                blocks.append((block, y, x))
                
                # 移动到下一个水平块
                if x + block_w >= w:
                    break
                x += block_size - overlap
            
            # 移动到下一个垂直块
            if y + block_h >= h:
                break
            y += block_size - overlap
        
        return blocks

    def _ocr_single_block(self, block: np.ndarray) -> List:
        """对单个图片块进行OCR识别"""
        result = self.ocr.ocr(block, cls=False)
        if result and result[0]:
            return result[0]
        return []

    def _merge_ocr_results(self, blocks_results: List[Tuple[List, int, int]]) -> List:
        """
        合并所有分块的OCR结果，并去重
        
        Args:
            blocks_results: [(ocr结果, y偏移量, x偏移量), ...]
            
        Returns:
            合并后的OCR结果
        """
        merged = []
        seen_texts = {}  # 用于去重: {(text, 近似y位置): result}
        
        for ocr_result, y_offset, x_offset in blocks_results:
            for line in ocr_result:
                box, (text, score) = line
                
                # 将坐标转换为原图坐标
                adjusted_box = [
                    [p[0] + x_offset, p[1] + y_offset] for p in box
                ]
                
                # 计算中心点用于去重
                center_y = sum(p[1] for p in adjusted_box) / 4
                center_x = sum(p[0] for p in adjusted_box) / 4
                
                # 使用文本内容和近似位置作为去重键
                # 位置精度为50像素，避免重叠区域的重复识别
                key = (text, int(center_y / 50), int(center_x / 50))
                
                if key not in seen_texts or seen_texts[key][1][1] < score:
                    seen_texts[key] = (adjusted_box, (text, score))
        
        merged = [(box, text_score) for box, text_score in seen_texts.values()]
        return merged

    def locate_keyword(
        self,
        image_path: str,
        keyword: str,
        draw: bool = True,
        output_path: str = "result.png",
        enable_split: bool = True,
        split_overlap: int = 100,
    ) -> List[Dict]:
        """
        定位关键词在图片中的位置
        
        Args:
            image_path: 图片路径
            keyword: 要查找的关键词
            draw: 是否绘制结果框
            output_path: 输出图片路径
            enable_split: 是否启用大图分块识别（默认启用）
            split_overlap: 分块重叠像素（默认100）
            
        Returns:
            匹配结果列表
        """
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        h, w = img.shape[:2]
        
        # 判断是否需要分块处理
        if enable_split and (h > self.det_limit_side_len or w > self.det_limit_side_len):
            # 大图分块处理
            blocks = self._split_image(img, overlap=split_overlap)
            blocks_results = []
            
            for block, y_offset, x_offset in blocks:
                block_result = self._ocr_single_block(block)
                if block_result:
                    blocks_results.append((block_result, y_offset, x_offset))
            
            ocr_lines = self._merge_ocr_results(blocks_results)
        else:
            # 小图直接处理
            ocr_result = self.ocr.ocr(image_path, cls=False)
            ocr_lines = ocr_result[0] if ocr_result and ocr_result[0] else []

        hits = []
        seen_boxes = set()

        for line in ocr_lines:
            box, (text, score) = line

            if keyword not in text:
                continue

            # 计算关键词在文本中的比例位置
            idx = text.find(keyword)
            ratio_start = idx / len(text)
            ratio_end = (idx + len(keyword)) / len(text)

            xs = [p[0] for p in box]
            ys = [p[1] for p in box]

            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)

            kw_x1 = int(x_min + (x_max - x_min) * ratio_start)
            kw_x2 = int(x_min + (x_max - x_min) * ratio_end)

            rect = (kw_x1, int(y_min), kw_x2, int(y_max))

            # 去重（防止 PaddleOCR 内部重复）
            if rect in seen_boxes:
                continue
            seen_boxes.add(rect)

            hits.append({
                "text": keyword,
                "confidence": float(score),
                "box": [
                    [kw_x1, int(y_min)],
                    [kw_x2, int(y_min)],
                    [kw_x2, int(y_max)],
                    [kw_x1, int(y_max)],
                ]
            })

            if draw:
                cv2.rectangle(
                    img,
                    (kw_x1, int(y_min)),
                    (kw_x2, int(y_max)),
                    (0, 0, 255),
                    2
                )

        if draw:
            cv2.imwrite(output_path, img)

        return hits
