# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: settings.py
# author: RPA高老师
# description: 项目配置管理

from dataclasses import dataclass, field
from typing import Tuple
import os


@dataclass
class OCRSettings:
    """OCR识别配置"""
    use_gpu: bool = True
    lang: str = "ch"
    det_limit_side_len: int = 960
    split_overlap: int = 100


@dataclass
class ExtractorSettings:
    """封面提取器配置"""
    bg_color_hex: str = "343434"
    color_tolerance: int = 15
    min_cover_area: int = 10000
    aspect_ratio_range: Tuple[float, float] = (0.5, 2.0)
    edge_crop: int = 10


@dataclass
class PathSettings:
    """路径配置"""
    input_dir: str = "data/input"
    output_dir: str = "data/output"
    covers_dir: str = "data/output/covers"
    debug_dir: str = "data/output/debug"


@dataclass
class Settings:
    """项目总配置"""
    ocr: OCRSettings = field(default_factory=OCRSettings)
    extractor: ExtractorSettings = field(default_factory=ExtractorSettings)
    paths: PathSettings = field(default_factory=PathSettings)
    
    def ensure_dirs(self):
        """确保所有目录存在"""
        os.makedirs(self.paths.input_dir, exist_ok=True)
        os.makedirs(self.paths.output_dir, exist_ok=True)
        os.makedirs(self.paths.covers_dir, exist_ok=True)
        os.makedirs(self.paths.debug_dir, exist_ok=True)


# 全局配置单例
_settings = None


def get_settings() -> Settings:
    """获取全局配置"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
