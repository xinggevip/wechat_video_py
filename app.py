# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: app.py
# author: RPA高老师
# description: FastAPI应用入口

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api import router
from src.config import get_settings

# 创建FastAPI应用
app = FastAPI(
    title="微信视频号封面提取API",
    description="基于OCR和视觉算法，从长截图中提取视频封面图片",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# 添加CORS中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(router)


@app.on_event("startup")
async def startup_event():
    """应用启动时执行"""
    settings = get_settings()
    settings.ensure_dirs()
    print("=" * 50)
    print("微信视频号封面提取API 已启动")
    print("=" * 50)
    print(f"API文档: http://localhost:8000/docs")
    print(f"ReDoc文档: http://localhost:8000/redoc")
    print("=" * 50)


@app.get("/", tags=["根路径"])
async def root():
    """API根路径"""
    return {
        "name": "微信视频号封面提取API",
        "version": "1.0.0",
        "docs": "/docs",
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
