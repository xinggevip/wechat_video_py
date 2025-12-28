# -*- coding: utf-8 -*-
# time: 2025/12/28
# file: test_api.py
# author: RPA高老师
# description: API接口测试

import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from src.config import get_settings


def test_health_check():
    """测试健康检查接口"""
    response = requests.get("http://localhost:8000/api/v1/health")
    print(f"[健康检查] 状态码: {response.status_code}")
    print(f"[健康检查] 响应: {response.json()}")
    assert response.status_code == 200


def test_locate_keyword():
    """测试关键词定位接口"""
    settings = get_settings()
    test_image = os.path.join(settings.paths.input_dir, "test_input.png")
    
    if not os.path.exists(test_image):
        print(f"[跳过] 测试图片不存在: {test_image}")
        return
    
    with open(test_image, "rb") as f:
        response = requests.post(
            "http://localhost:8000/api/v1/locate-keyword",
            files={"image": ("test.png", f, "image/png")},
            data={"keyword": "动态"},
        )
    
    print(f"[关键词定位] 状态码: {response.status_code}")
    print(f"[关键词定位] 响应: {response.json()}")


def test_extract_covers():
    """测试封面提取接口"""
    settings = get_settings()
    test_image = os.path.join(settings.paths.input_dir, "test_input.png")
    
    if not os.path.exists(test_image):
        print(f"[跳过] 测试图片不存在: {test_image}")
        return
    
    with open(test_image, "rb") as f:
        response = requests.post(
            "http://localhost:8000/api/v1/extract-covers",
            files={"image": ("test.png", f, "image/png")},
            data={
                "keyword": "动态",
                "bg_color": "343434",
                "edge_crop": "10",
            },
        )
    
    print(f"[封面提取] 状态码: {response.status_code}")
    
    if response.status_code == 200:
        # 保存返回的ZIP文件
        output_path = os.path.join(settings.paths.output_dir, "api_test_covers.zip")
        with open(output_path, "wb") as f:
            f.write(response.content)
        print(f"[封面提取] ZIP文件已保存到: {output_path}")
    else:
        print(f"[封面提取] 错误: {response.text}")


if __name__ == "__main__":
    print("=" * 50)
    print("API接口测试")
    print("请确保API服务已启动: python app.py")
    print("=" * 50)
    
    try:
        test_health_check()
        print()
        test_locate_keyword()
        print()
        test_extract_covers()
    except requests.exceptions.ConnectionError:
        print("[错误] 无法连接到API服务，请先启动服务: python app.py")
