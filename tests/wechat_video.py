# -*- coding: utf-8 -*-
# time: 2025/12/28 09:57
# file: wechat_video.py
# author: RPA高老师

import keyword
import os
import shutil
import zipfile
import requests


def get_desktop_path():
    """获取用户桌面路径"""
    return os.path.join(os.path.expanduser("~"), "Desktop")


def get_video_cover_images_by_api(long_screenshot_img_path, keyword):
    """
    调用API获取视频封面图片
    
    Args:
        long_screenshot_img_path: 长截图图片路径
        
    Returns:
        list: 解压后所有文件的绝对路径列表
    """
    # 1. 获取桌面路径下的"微信小视频"文件夹
    desktop_path = get_desktop_path()
    target_folder = os.path.join(desktop_path, "微信小视频", keyword, "视频封面")
    
    # 2. 若没有该文件夹则创建，若有则清空
    if os.path.exists(target_folder):
        # 清空文件夹
        shutil.rmtree(target_folder)
    os.makedirs(target_folder)
    
    # 3. 调用API获取zip文件
    url = "http://localhost:8000/api/v1/extract-covers"
    file_name = os.path.basename(long_screenshot_img_path)
    
    payload = {'keyword': '动态'}
    with open(long_screenshot_img_path, 'rb') as f:
        files = [
            ('image', (file_name, f, 'image/png'))
        ]
        headers = {}
        response = requests.request("POST", url, headers=headers, data=payload, files=files)
    
    # 检查响应状态
    if response.status_code != 200:
        raise Exception(f"API请求失败，状态码: {response.status_code}")
    
    # 4. 下载zip文件并重命名为video_cover_images.zip
    zip_file_path = os.path.join(target_folder, "video_cover_images.zip")
    with open(zip_file_path, 'wb') as f:
        f.write(response.content)
    
    
    # 6. 解压zip文件到临时文件夹
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(target_folder)
    
    # 删除zip_file_path
    os.remove(zip_file_path)
    
    # 7. 获取临时文件夹里所有文件的绝对路径
    file_paths = []
    for root, dirs, files in os.walk(target_folder):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    
    return file_paths


if __name__ == "__main__":
    long_screenshot_img_path = r"C:\Users\Admin\Desktop\temp_screenshot_img.png"
    keyword = '途虎养车'
    cover_images = get_video_cover_images_by_api(long_screenshot_img_path, keyword)
    
    print(cover_images)
