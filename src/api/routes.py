# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: routes.py
# author: RPA高老师
# description: Web API路由定义

import os
import uuid
import shutil
import zipfile
import tempfile
from io import BytesIO
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse

from src.ocr import OCRRunner
from src.extractor import CoverExtractorCV
from src.config import get_settings

router = APIRouter(prefix="/api/v1", tags=["封面提取"])

# OCR实例（全局单例，避免重复加载模型）
_ocr_instance = None


def get_ocr() -> OCRRunner:
    """获取OCR实例（懒加载单例）"""
    global _ocr_instance
    if _ocr_instance is None:
        settings = get_settings()
        _ocr_instance = OCRRunner(
            use_gpu=settings.ocr.use_gpu,
            lang=settings.ocr.lang,
            det_limit_side_len=settings.ocr.det_limit_side_len,
        )
    return _ocr_instance


@router.post("/extract-covers", summary="提取视频封面")
async def extract_covers(
    image: UploadFile = File(..., description="长截图图片文件"),
    keyword: str = Form(default="动态", description="定位关键词"),
    bg_color: str = Form(default="343434", description="背景色HEX值"),
    color_tolerance: int = Form(default=15, description="颜色容差"),
    edge_crop: int = Form(default=10, description="裁剪封面四周的像素"),
):
    """
    从长截图中提取视频封面
    
    - **image**: 上传的长截图图片文件
    - **keyword**: 用于定位封面起始位置的关键词（默认：动态）
    - **bg_color**: 背景色HEX值（默认：343434）
    - **color_tolerance**: 颜色容差（默认：15）
    - **edge_crop**: 裁剪封面四周的像素（默认：10）
    
    返回包含所有封面图片的ZIP压缩包
    """
    # 生成唯一的临时目录
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"cover_extract_{request_id}")
    
    try:
        # 创建临时目录
        os.makedirs(temp_dir, exist_ok=True)
        covers_dir = os.path.join(temp_dir, "covers")
        os.makedirs(covers_dir, exist_ok=True)
        
        # 保存上传的图片
        input_image_path = os.path.join(temp_dir, "input_image.png")
        with open(input_image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # 1. 使用OCR定位关键词
        ocr = get_ocr()
        results = ocr.locate_keyword(
            image_path=input_image_path,
            keyword=keyword,
            draw=False,
        )
        
        if not results:
            raise HTTPException(
                status_code=400,
                detail=f"未在图片中找到关键词 '{keyword}'"
            )
        
        keyword_box = results[0]['box']
        
        # 2. 提取封面
        settings = get_settings()
        extractor = CoverExtractorCV(
            bg_color_hex=bg_color,
            color_tolerance=color_tolerance,
            min_cover_area=settings.extractor.min_cover_area,
            aspect_ratio_range=settings.extractor.aspect_ratio_range,
            edge_crop=edge_crop,
        )
        
        extracted_paths = extractor.extract_covers(
            image_path=input_image_path,
            keyword_box=keyword_box,
            output_dir=covers_dir,
            debug=False,
        )
        
        if not extracted_paths:
            raise HTTPException(
                status_code=400,
                detail="未能提取到任何封面图片"
            )
        
        # 3. 创建ZIP压缩包
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for cover_path in extracted_paths:
                # 只使用文件名作为压缩包内的路径
                arcname = os.path.basename(cover_path)
                zip_file.write(cover_path, arcname)
        
        zip_buffer.seek(0)
        
        # 4. 返回ZIP文件
        return StreamingResponse(
            zip_buffer,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=covers_{request_id[:8]}.zip"
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.post("/locate-keyword", summary="定位关键词")
async def locate_keyword(
    image: UploadFile = File(..., description="图片文件"),
    keyword: str = Form(..., description="要定位的关键词"),
):
    """
    在图片中定位关键词位置
    
    - **image**: 上传的图片文件
    - **keyword**: 要定位的关键词
    
    返回关键词的坐标信息
    """
    request_id = str(uuid.uuid4())
    temp_dir = os.path.join(tempfile.gettempdir(), f"keyword_locate_{request_id}")
    
    try:
        os.makedirs(temp_dir, exist_ok=True)
        
        # 保存上传的图片
        input_image_path = os.path.join(temp_dir, "input_image.png")
        with open(input_image_path, "wb") as f:
            content = await image.read()
            f.write(content)
        
        # 使用OCR定位关键词
        ocr = get_ocr()
        results = ocr.locate_keyword(
            image_path=input_image_path,
            keyword=keyword,
            draw=False,
        )
        
        return {
            "success": True,
            "keyword": keyword,
            "count": len(results),
            "results": results,
        }
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"处理失败: {str(e)}"
        )
    finally:
        # 清理临时目录
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@router.get("/health", summary="健康检查")
async def health_check():
    """API健康检查"""
    return {"status": "ok", "message": "服务运行正常"}
