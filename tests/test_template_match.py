# -*- coding: utf-8 -*-
# 测试模板匹配功能

import cv2
import numpy as np
import sys
sys.path.insert(0, 'e:/project/myself/python/weixin_video')

from src.ocr.ocr_runner import find_template, find_all_templates


def test_with_real_images():
    """使用真实图片测试（请替换为你的图片路径）"""
    # 读取全屏截图和模板图片
    screenshot = cv2.imread(r'C:\Users\Admin\Desktop\全屏截图.png')
    template = cv2.imread(r'C:\Users\Admin\Desktop\微信小视频\迪士尼\视频封面\2.png')
    
    if screenshot is None:
        print("❌ 请先准备全屏截图：data/input/fullscreen.png")
        return
    if template is None:
        print("❌ 请先准备模板图片：data/input/template.png")
        return
    
    # 执行匹配
    found, (x, y, w, h), confidence = find_template(screenshot, template, threshold=0.8)
    
    if found:
        print(f"✅ 找到目标！")
        print(f"   位置: ({x}, {y})")
        print(f"   大小: {w} x {h}")
        print(f"   置信度: {confidence:.4f}")
        
        # 在截图上画框并保存结果
        result_img = screenshot.copy()
        cv2.rectangle(result_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.imwrite('e:/project/myself/python/weixin_video/data/input/result.png', result_img)
        print(f"   结果已保存到: data/input/result.png")
    else:
        print(f"❌ 未找到匹配，最高置信度: {confidence:.4f}")


def test_with_synthetic_images():
    """使用合成图片测试（无需准备图片）"""
    print("=== 合成图片测试 ===")
    
    # 创建一个模拟的全屏截图 (500x400 灰色背景)
    screenshot = np.ones((400, 500, 3), dtype=np.uint8) * 128
    
    # 在截图中画一个红色矩形作为"目标"
    cv2.rectangle(screenshot, (150, 100), (250, 180), (0, 0, 255), -1)
    
    # 创建模板（从截图中截取红色矩形区域）
    template = screenshot[100:180, 150:250].copy()
    
    # 测试查找
    found, (x, y, w, h), confidence = find_template(screenshot, template, threshold=0.9)
    
    print(f"找到: {found}")
    print(f"位置: ({x}, {y}), 大小: {w}x{h}")
    print(f"置信度: {confidence:.4f}")
    
    # 验证结果
    assert found == True, "应该找到匹配"
    assert x == 150 and y == 100, f"位置应为(150, 100)，实际({x}, {y})"
    print("✅ 合成图片测试通过！\n")


def test_no_match():
    """测试找不到的情况"""
    print("=== 无匹配测试 ===")
    
    # 创建两个完全不同的图片
    screenshot = np.ones((400, 500, 3), dtype=np.uint8) * 200  # 浅灰色
    template = np.zeros((50, 50, 3), dtype=np.uint8)  # 黑色
    
    found, box, confidence = find_template(screenshot, template, threshold=0.9)
    
    print(f"找到: {found}")
    print(f"置信度: {confidence:.4f}")
    assert found == False, "不应该找到匹配"
    print("✅ 无匹配测试通过！\n")


if __name__ == '__main__':
    # 先运行合成图片测试（验证函数正确性）
    # test_with_synthetic_images()
    # test_no_match()
    
    # 然后用真实图片测试
    print("=== 真实图片测试 ===")
    test_with_real_images()
