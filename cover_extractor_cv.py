# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: cover_extractor_cv.py
# author: RPA高老师
# description: 使用纯视觉算法，基于背景色识别并提取封面图片

import os
import cv2
import numpy as np
from typing import List, Tuple, Optional


class CoverExtractorCV:
    """基于视觉算法的封面提取器"""
    
    def __init__(
        self,
        bg_color_hex: str = "343434",         # 背景色 HEX 值
        color_tolerance: int = 15,             # 颜色容差
        min_cover_area: int = 10000,           # 最小封面面积（过滤噪点）
        aspect_ratio_range: Tuple[float, float] = (0.5, 2.0),  # 宽高比范围
        edge_crop: int = 0,                    # 裁剪图片四周的像素
    ):
        """
        初始化封面提取器
        
        Args:
            bg_color_hex: 背景色的HEX值（不含#）
            color_tolerance: 颜色容差，用于匹配背景色
            min_cover_area: 最小封面面积，过滤小区域
            aspect_ratio_range: 封面宽高比范围，过滤异常形状
            edge_crop: 裁剪图片四周的像素，如传入10则上下左右各裁掉10px
        """
        self.bg_color_hex = bg_color_hex
        self.color_tolerance = color_tolerance
        self.min_cover_area = min_cover_area
        self.aspect_ratio_range = aspect_ratio_range
        self.edge_crop = edge_crop
        
        # 解析背景色 HEX -> BGR
        self.bg_color_bgr = self._hex_to_bgr(bg_color_hex)
    
    def _hex_to_bgr(self, hex_color: str) -> Tuple[int, int, int]:
        """HEX颜色转BGR"""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return (b, g, r)  # OpenCV使用BGR格式
    
    def _create_background_mask(self, image: np.ndarray) -> np.ndarray:
        """
        创建背景掩码，背景为白色(255)，前景为黑色(0)
        """
        # 计算每个像素与背景色的差异
        bg_color = np.array(self.bg_color_bgr, dtype=np.int32)
        diff = np.abs(image.astype(np.int32) - bg_color)
        max_diff = np.max(diff, axis=2)
        
        # 差异小于容差的视为背景
        bg_mask = (max_diff <= self.color_tolerance).astype(np.uint8) * 255
        
        return bg_mask
    
    def _find_cover_contours(self, fg_mask: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        在前景掩码中找到封面的边界框
        
        Returns:
            封面边界框列表 [(x, y, w, h), ...]
        """
        # 形态学操作，填充小孔洞，连接相近区域
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        
        # 查找轮廓
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        cover_boxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            
            # 过滤条件
            if area < self.min_cover_area:
                continue
            
            aspect_ratio = w / h if h > 0 else 0
            if not (self.aspect_ratio_range[0] <= aspect_ratio <= self.aspect_ratio_range[1]):
                continue
            
            cover_boxes.append((x, y, w, h))
        
        return cover_boxes
    
    def _standardize_cover_size(self, boxes: List[Tuple[int, int, int, int]]) -> Tuple[int, int]:
        """
        根据检测到的封面，确定标准尺寸（使用众数或中位数）
        """
        if not boxes:
            return (0, 0)
        
        widths = [box[2] for box in boxes]
        heights = [box[3] for box in boxes]
        
        # 使用中位数作为标准尺寸
        std_width = int(np.median(widths))
        std_height = int(np.median(heights))
        
        return (std_width, std_height)
    
    def extract_covers(
        self,
        image_path: str,
        keyword_box: Optional[List[List[int]]] = None,
        output_dir: str = "covers",
        debug: bool = False,
    ) -> List[str]:
        """
        提取封面图片
        
        Args:
            image_path: 输入图片路径
            keyword_box: 关键词的box坐标，从该坐标下方开始提取。None则从图片顶部开始
            output_dir: 输出目录
            debug: 是否保存调试图片
            
        Returns:
            提取的封面图片路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 读取图片
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
        
        img_height, img_width = image.shape[:2]
        print(f"图片尺寸: {img_width} x {img_height}")
        
        # 确定起始Y坐标
        start_y = 0
        if keyword_box:
            # 使用关键词box的底部作为起始位置
            start_y = max(keyword_box[2][1], keyword_box[3][1])
            print(f"从关键词底部开始提取，起始Y坐标: {start_y}")
        
        # 裁剪关键词以下的区域
        roi = image[start_y:, :]
        
        # 创建背景掩码
        bg_mask = self._create_background_mask(roi)
        
        # 前景掩码（非背景区域）
        fg_mask = cv2.bitwise_not(bg_mask)
        
        if debug:
            cv2.imwrite(os.path.join(output_dir, "_debug_bg_mask.png"), bg_mask)
            cv2.imwrite(os.path.join(output_dir, "_debug_fg_mask.png"), fg_mask)
        
        # 查找封面轮廓
        cover_boxes = self._find_cover_contours(fg_mask)
        print(f"检测到 {len(cover_boxes)} 个候选封面区域")
        
        if not cover_boxes:
            print("未检测到封面区域")
            return []
        
        # 确定标准封面尺寸
        std_width, std_height = self._standardize_cover_size(cover_boxes)
        print(f"标准封面尺寸: {std_width} x {std_height}")
        
        # 按位置排序：先按Y（行），再按X（列）
        # 使用标准高度的一半作为行分组阈值
        row_threshold = std_height // 2
        
        def sort_key(box):
            x, y, w, h = box
            # 将Y坐标量化到行
            row = y // row_threshold
            return (row, x)
        
        cover_boxes.sort(key=sort_key)
        
        # 过滤尺寸异常的封面（与标准尺寸差异过大的）
        size_tolerance = 0.3  # 30%容差
        filtered_boxes = []
        for box in cover_boxes:
            x, y, w, h = box
            width_ok = abs(w - std_width) / std_width <= size_tolerance if std_width > 0 else True
            height_ok = abs(h - std_height) / std_height <= size_tolerance if std_height > 0 else True
            if width_ok and height_ok:
                filtered_boxes.append(box)
            else:
                print(f"过滤异常尺寸封面: 位置({x}, {y}), 尺寸({w}x{h})")
        
        print(f"过滤后剩余 {len(filtered_boxes)} 个封面")
        
        # 提取并保存封面
        extracted_paths = []
        for i, (x, y, w, h) in enumerate(filtered_boxes, 1):
            # 从ROI中裁剪封面
            cover = roi[y:y+h, x:x+w]
            
            # 裁剪四周边缘
            if self.edge_crop > 0:
                crop = self.edge_crop
                cover_h, cover_w = cover.shape[:2]
                # 确保裁剪后仍有有效区域
                if cover_h > crop * 2 and cover_w > crop * 2:
                    cover = cover[crop:cover_h-crop, crop:cover_w-crop]
            
            # 保存
            output_path = os.path.join(output_dir, f"{i}.png")
            cv2.imwrite(output_path, cover)
            extracted_paths.append(output_path)
            
            print(f"提取封面 #{i}: 位置({x}, {y + start_y}), 尺寸({w}x{h})")
        
        # 保存调试图片（标注检测结果）
        if debug:
            debug_image = image.copy()
            for i, (x, y, w, h) in enumerate(filtered_boxes, 1):
                # 调整坐标到原图
                cv2.rectangle(debug_image, (x, y + start_y), (x + w, y + h + start_y), (0, 255, 0), 2)
                cv2.putText(debug_image, str(i), (x + 5, y + start_y + 25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(os.path.join(output_dir, "_debug_detected.png"), debug_image)
        
        print(f"\n共提取 {len(extracted_paths)} 个封面到目录: {output_dir}")
        return extracted_paths


def main():
    """示例用法"""
    from ocr.ocr_runner import OCRRunner
    
    # 1. 初始化OCR
    ocr = OCRRunner(
        use_gpu=True,
        lang="ch",
        det_limit_side_len=960,
    )
    
    # 2. 定位"动态"关键词
    image_path = r"src/input.png"
    results = ocr.locate_keyword(
        image_path=image_path,
        keyword="动态",
        draw=False,
    )
    
    if not results:
        print("未找到'动态'关键词")
        return
    
    print(f"找到 {len(results)} 个'动态'关键词")
    keyword_box = results[0]['box']
    print(f"使用的box坐标: {keyword_box}")
    
    # 3. 初始化视觉封面提取器
    extractor = CoverExtractorCV(
        bg_color_hex="343434",       # 背景色
        color_tolerance=10,           # 颜色容差
        min_cover_area=10000,         # 最小封面面积
        aspect_ratio_range=(0.5, 2.0),  # 宽高比范围
        edge_crop=10,                 # 裁剪四周10px
    )
    
    # 4. 提取封面
    extracted = extractor.extract_covers(
        image_path=image_path,
        keyword_box=keyword_box,
        output_dir="covers",
        debug=True,  # 保存调试图片
    )
    
    print(f"\n提取完成！共 {len(extracted)} 个封面")


if __name__ == "__main__":
    main()
