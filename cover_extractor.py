# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: cover_extractor.py
# author: RPA高老师
# description: 根据OCR识别的关键词坐标，按规律抠出封面图片

import os
from PIL import Image
from typing import List, Tuple, Optional


class CoverExtractor:
    """封面提取器"""
    
    def __init__(
        self,
        keyword_to_cover_gap: int = 10,      # 关键词距离下面封面的距离
        side_margin: int = 150,               # 封面整体内容距离两边的宽度
        cover_horizontal_gap: int = 14,       # 每个封面中间的水平间距
        cover_width: int = 408,               # 每个封面的宽度
        cover_height: int = 310,              # 每个封面的高度
        cover_vertical_gap: int = 14,         # 每个封面的垂直间距（行与行之间）
        covers_per_row: int = 6,              # 每行封面数量
    ):
        """
        初始化封面提取器
        
        Args:
            keyword_to_cover_gap: "动态"关键词距离下面封面的距离（像素）
            side_margin: 封面整体内容距离两边的宽度（像素）
            cover_horizontal_gap: 每个封面中间的水平间距（像素）
            cover_width: 每个封面的宽度（像素）
            cover_height: 每个封面的高度（像素）
            cover_vertical_gap: 每个封面的垂直间距（像素）
            covers_per_row: 每行封面数量
        """
        self.keyword_to_cover_gap = keyword_to_cover_gap
        self.side_margin = side_margin
        self.cover_horizontal_gap = cover_horizontal_gap
        self.cover_width = cover_width
        self.cover_height = cover_height
        self.cover_vertical_gap = cover_vertical_gap
        self.covers_per_row = covers_per_row
    
    def extract_covers(
        self,
        image_path: str,
        keyword_box: List[List[int]],
        output_dir: str = "covers",
        max_covers: Optional[int] = None,
    ) -> List[str]:
        """
        根据关键词box坐标，提取所有封面图片
        
        Args:
            image_path: 输入图片路径
            keyword_box: 关键词的box坐标，格式为 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            output_dir: 输出目录
            max_covers: 最大提取封面数量，None表示提取所有可能的封面
            
        Returns:
            提取的封面图片路径列表
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 打开图片
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 计算封面区域的起始Y坐标（关键词box的底部 + 间距）
        # box格式: [[左上], [右上], [右下], [左下]]
        keyword_bottom = max(keyword_box[2][1], keyword_box[3][1])
        start_y = keyword_bottom + self.keyword_to_cover_gap
        
        # 使用配置的封面宽度
        cover_width = self.cover_width
        
        print(f"图片尺寸: {img_width} x {img_height}")
        print(f"封面起始Y坐标: {start_y}")
        print(f"封面宽度: {cover_width}")
        print(f"封面高度: {self.cover_height}")
        
        extracted_paths = []
        cover_index = 1
        current_y = start_y
        
        # 逐行提取封面
        while current_y + self.cover_height <= img_height:
            # 检查是否达到最大封面数量
            if max_covers and cover_index > max_covers:
                break
                
            # 提取当前行的每个封面
            for col in range(self.covers_per_row):
                if max_covers and cover_index > max_covers:
                    break
                
                # 计算当前封面的X坐标
                x = self.side_margin + col * (cover_width + self.cover_horizontal_gap)
                
                # 确保不超出图片边界
                if x + cover_width > img_width:
                    break
                
                # 裁剪封面
                box = (x, current_y, x + cover_width, current_y + self.cover_height)
                cover = img.crop(box)
                
                # 保存封面
                output_path = os.path.join(output_dir, f"{cover_index}.png")
                cover.save(output_path)
                extracted_paths.append(output_path)
                
                print(f"提取封面 #{cover_index}: 位置({x}, {current_y}), 尺寸({cover_width}x{self.cover_height})")
                cover_index += 1
            
            # 移动到下一行
            current_y += self.cover_height + self.cover_vertical_gap
        
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
    
    # 3. 使用第一个匹配结果的box
    keyword_box = results[0]['box']
    print(f"使用的box坐标: {keyword_box}")
    
    # 4. 初始化封面提取器（可自定义参数）
    extractor = CoverExtractor(
        keyword_to_cover_gap=12,      # 关键词距离封面10px
        side_margin=152,               # 左右边距150px
        cover_horizontal_gap=14,       # 封面水平间距14px
        cover_width=255,               # 封面宽度408px
        cover_height=340,              # 封面高度370px
        cover_vertical_gap=68,         # 封面垂直间距14px
        covers_per_row=6,              # 每行6个封面
    )
    
    # 5. 提取封面
    output_dir = "covers"  # 可自定义输出目录
    extracted = extractor.extract_covers(
        image_path=image_path,
        keyword_box=keyword_box,
        output_dir=output_dir,
        max_covers=None,  # 提取所有封面，可设置具体数字限制
    )
    
    print(f"\n提取完成！共 {len(extracted)} 个封面")


if __name__ == "__main__":
    main()
