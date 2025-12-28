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


def find_template(
    screenshot: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8
) -> Tuple[bool, Tuple[int, int, int, int], float]:
    """
    从全屏截图中找到模板图片的位置
    
    Args:
        screenshot: 全屏截图（BGR格式的numpy数组）
        template: 要查找的模板图片（屏幕中某一块的截图）
        threshold: 匹配相似度阈值，范围0-1，默认0.8
        
    Returns:
        Tuple[bool, Tuple[int, int, int, int], float]:
            - found: 是否找到匹配
            - box: 匹配区域的坐标 (x, y, width, height)，未找到时为 (0, 0, 0, 0)
            - confidence: 匹配置信度
            
    Example:
        >>> screenshot = cv2.imread('fullscreen.png')
        >>> template = cv2.imread('button.png')
        >>> found, (x, y, w, h), confidence = find_template(screenshot, template, 0.9)
        >>> if found:
        ...     print(f'找到目标，位置: ({x}, {y})，大小: {w}x{h}，置信度: {confidence:.2f}')
    """
    if screenshot is None or template is None:
        return False, (0, 0, 0, 0), 0.0
    
    # 获取模板尺寸
    h, w = template.shape[:2]
    
    # 确保截图尺寸大于等于模板尺寸
    if screenshot.shape[0] < h or screenshot.shape[1] < w:
        return False, (0, 0, 0, 0), 0.0
    
    # 转换为灰度图进行匹配（提高速度和准确性）
    if len(screenshot.shape) == 3:
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    else:
        screenshot_gray = screenshot
        
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    # 使用归一化相关系数匹配法
    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 获取最佳匹配位置和值
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    
    # 判断是否达到阈值
    if max_val >= threshold:
        x, y = max_loc
        return True, (x, y, w, h), float(max_val)
    
    return False, (0, 0, 0, 0), float(max_val)


def find_all_templates(
    screenshot: np.ndarray,
    template: np.ndarray,
    threshold: float = 0.8
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    从全屏截图中找到所有匹配模板图片的位置
    
    Args:
        screenshot: 全屏截图（BGR格式的numpy数组）
        template: 要查找的模板图片
        threshold: 匹配相似度阈值，范围0-1，默认0.8
        
    Returns:
        List[Tuple[Tuple[int, int, int, int], float]]:
            匹配结果列表，每个元素为 ((x, y, width, height), confidence)
            
    Example:
        >>> screenshot = cv2.imread('fullscreen.png')
        >>> template = cv2.imread('icon.png')
        >>> matches = find_all_templates(screenshot, template, 0.85)
        >>> for (x, y, w, h), conf in matches:
        ...     print(f'位置: ({x}, {y})，置信度: {conf:.2f}')
    """
    if screenshot is None or template is None:
        return []
    
    h, w = template.shape[:2]
    
    if screenshot.shape[0] < h or screenshot.shape[1] < w:
        return []
    
    # 转换为灰度图
    if len(screenshot.shape) == 3:
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)
    else:
        screenshot_gray = screenshot
        
    if len(template.shape) == 3:
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template
    
    result = cv2.matchTemplate(screenshot_gray, template_gray, cv2.TM_CCOEFF_NORMED)
    
    # 找到所有超过阈值的位置
    locations = np.where(result >= threshold)
    
    matches = []
    for pt in zip(*locations[::-1]):  # 转换为 (x, y) 格式
        x, y = pt
        confidence = float(result[y, x])
        matches.append(((x, y, w, h), confidence))
    
    # 使用非极大值抑制去除重叠的匹配
    if matches:
        matches = _non_max_suppression(matches, w, h)
    
    # 按置信度降序排序
    matches.sort(key=lambda m: m[1], reverse=True)
    
    return matches


def _non_max_suppression(
    matches: List[Tuple[Tuple[int, int, int, int], float]],
    w: int,
    h: int,
    overlap_thresh: float = 0.5
) -> List[Tuple[Tuple[int, int, int, int], float]]:
    """
    非极大值抑制，去除重叠的匹配结果
    """
    if not matches:
        return []
    
    # 按置信度排序
    matches = sorted(matches, key=lambda m: m[1], reverse=True)
    
    keep = []
    for match in matches:
        (x, y, _, _), conf = match
        
        # 检查是否与已保留的匹配重叠
        is_overlap = False
        for kept_match in keep:
            (kx, ky, _, _), _ = kept_match
            # 如果中心点距离小于宽高的一半，认为重叠
            if abs(x - kx) < w * overlap_thresh and abs(y - ky) < h * overlap_thresh:
                is_overlap = True
                break
        
        if not is_overlap:
            keep.append(match)
    
    return keep
