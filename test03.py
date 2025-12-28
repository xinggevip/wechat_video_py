# -*- coding: utf-8 -*-
# time: 2025/12/27 16:10
# file: test03.py
# author: RPA高老师


import os
import numpy as np
import paddleocr
from paddleocr import PaddleOCR
import cv2
from PIL import Image
import traceback  # 用于输出详细异常信息（含行号、堆栈）

# 初始化PaddleOCR模型（纯CPU模式）
ocr = PaddleOCR(
    use_angle_cls=True,
    use_gpu=False,
    lang="ch",
    show_log=False
)

# 大图片缩放配置（可根据需求调整）
MAX_IMAGE_WIDTH = 2000  # 最大宽度限制，超出则等比例缩放
MAX_IMAGE_HEIGHT = 3000  # 最大高度限制，超出则等比例缩放（适配长截图）

def resize_large_image(image_np: np.ndarray, max_w: int, max_h: int) -> tuple[np.ndarray, float, float]:
    """
    超大图片等比例缩放预处理
    :param image_np: 原始OpenCV格式图片
    :param max_w: 最大宽度
    :param max_h: 最大高度
    :return: 缩放后的图片、宽度缩放因子、高度缩放因子
    """
    # 先校验输入图片是否为None
    if image_np is None:
        return np.array([]), 1.0, 1.0

    # 获取原始图片尺寸
    h, w = image_np.shape[:2]
    # 计算缩放因子（宽高分别计算，取最小因子保证不超出限制，等比例缩放）
    scale_w = max_w / w if w > max_w else 1.0
    scale_h = max_h / h if h > max_h else 1.0
    scale_factor = min(scale_w, scale_h)

    # 无需缩放的情况
    if scale_factor == 1.0:
        return image_np, 1.0, 1.0

    # 计算缩放后的尺寸
    new_w = int(w * scale_factor)
    new_h = int(h * scale_factor)
    # 等比例缩放图片（使用cv2.INTER_AREA，对缩小图片效果最优）
    resized_image = cv2.resize(
        src=image_np,
        dsize=(new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    return resized_image, scale_factor, scale_factor  # 宽高缩放因子一致（等比例）

def get_keyword_coordinates(image_np: np.ndarray, target_text: str, ignore_case: bool = True) -> tuple[list, float, float]:
    """
    核心功能：精准获取关键字坐标（支持超大图片，返回缩放因子用于坐标还原）
    :param image_np: 原始OpenCV格式图片
    :param target_text: 目标关键字
    :param ignore_case: 是否忽略大小写
    :return: 关键字结果列表、宽度缩放因子、高度缩放因子
    """
    keyword_results = []
    # 1. 校验原始图片是否有效
    if image_np is None or len(image_np.shape) == 0:
        return keyword_results, 1.0, 1.0

    # 2. 超大图片预处理：等比例缩放
    resized_image, scale_w, scale_h = resize_large_image(
        image_np=image_np,
        max_w=MAX_IMAGE_WIDTH,
        max_h=MAX_IMAGE_HEIGHT
    )

    # 3. 校验缩放后的图片是否有效
    if resized_image is None or len(resized_image.shape) == 0:
        return keyword_results, scale_w, scale_h

    # 4. 对缩放后的图片执行OCR识别（先判断返回结果是否为None，避免迭代报错）
    ocr_results = ocr.ocr(resized_image, cls=True)
    # 关键修复：先判断ocr_results不是None，再判断是否为空
    if ocr_results is None:
        print("⚠️ OCR识别返回None，未获取到文本信息")
        return keyword_results, scale_w, scale_h
    if not ocr_results:
        return keyword_results, scale_w, scale_h

    target_len = len(target_text)
    if target_len == 0:
        return keyword_results, scale_w, scale_h

    # 5. 遍历文本块，精准定位关键字（增加各层级非空判断）
    for page_idx, page in enumerate(ocr_results):
        # 校验page是否为None
        if page is None:
            continue
        for text_area in page:
            # 校验text_area是否为None
            if text_area is None or len(text_area) < 2:
                continue
            # 提取文本块信息
            text_box = text_area[0]  # 缩放后文本块的四顶点坐标
            recognized_text = text_area[1][0]  # 整段识别文字
            confidence = float(text_area[1][1])  # 置信度
            # 校验文本内容是否有效
            if not recognized_text:
                continue
            text_len = len(recognized_text)

            # 匹配关键字，获取起始索引
            match_indexes = []
            if ignore_case:
                target_match = target_text.lower()
                recognized_match = recognized_text.lower()
            else:
                target_match = target_text
                recognized_match = recognized_text

            start_idx = 0
            while start_idx <= text_len - target_len:
                idx = recognized_match.find(target_match, start_idx)
                if idx == -1:
                    break
                match_indexes.append(idx)
                start_idx = idx + target_len

            if not match_indexes:
                continue

            # 计算缩放后文本块的外接矩形
            xmin_block = int(min([pt[0] for pt in text_box]))
            ymin_block = int(min([pt[1] for pt in text_box]))
            xmax_block = int(max([pt[0] for pt in text_box]))
            ymax_block = int(max([pt[1] for pt in text_box]))
            block_width = xmax_block - xmin_block
            block_height = ymax_block - ymin_block
            char_avg_width = block_width / text_len if text_len > 0 else 0

            # 计算关键字缩放后的精准坐标
            for idx in match_indexes:
                keyword_start_x = xmin_block + idx * char_avg_width
                keyword_end_x = xmin_block + (idx + target_len) * char_avg_width
                # 缩放后的关键字坐标
                keyword_xmin_resized = int(round(keyword_start_x))
                keyword_ymin_resized = ymin_block
                keyword_xmax_resized = int(round(keyword_end_x))
                keyword_ymax_resized = ymax_block

                # 整理缩放后的结果（后续将还原为原始坐标）
                keyword_results.append({
                    "page_index": page_idx,
                    "target_keyword": target_text,
                    "recognized_text_block": recognized_text,
                    "keyword_position_in_block": idx,
                    "confidence": confidence,
                    "keyword_bounding_box_resized": [keyword_xmin_resized, keyword_ymin_resized, keyword_xmax_resized, keyword_ymax_resized],
                    "text_block_bounding_box_resized": [xmin_block, ymin_block, xmax_block, ymax_block]
                })

    return keyword_results, scale_w, scale_h

def restore_original_coordinates(keyword_results: list, scale_w: float, scale_h: float) -> list:
    """
    将缩放后的关键字坐标还原为原始图片的真实坐标
    :param keyword_results: 缩放后的关键字结果
    :param scale_w: 宽度缩放因子
    :param scale_h: 高度缩放因子
    :return: 原始坐标的关键字结果
    """
    original_results = []
    # 校验keyword_results是否为None
    if keyword_results is None:
        return original_results

    for result in keyword_results:
        # 还原关键字精准坐标
        xmin_resized, ymin_resized, xmax_resized, ymax_resized = result["keyword_bounding_box_resized"]
        xmin_original = int(round(xmin_resized / scale_w))
        ymin_original = int(round(ymin_resized / scale_h))
        xmax_original = int(round(xmax_resized / scale_w))
        ymax_original = int(round(ymax_resized / scale_h))

        # 还原文本块坐标（参考用）
        bxmin_resized, bymin_resized, bxmax_resized, bymax_resized = result["text_block_bounding_box_resized"]
        bxmin_original = int(round(bxmin_resized / scale_w))
        bymin_original = int(round(bymin_resized / scale_h))
        bxmax_original = int(round(bxmax_resized / scale_w))
        bymax_original = int(round(bymax_resized / scale_h))

        # 整理原始坐标结果
        original_result = {
            "page_index": result["page_index"],
            "target_keyword": result["target_keyword"],
            "recognized_text_block": result["recognized_text_block"],
            "keyword_position_in_block": result["keyword_position_in_block"],
            "confidence": result["confidence"],
            "keyword_bounding_box": [xmin_original, ymin_original, xmax_original, ymax_original],  # 原始精准坐标
            "text_block_bounding_box": [bxmin_original, bymin_original, bxmax_original, bymax_original]  # 原始文本块坐标
        }
        original_results.append(original_result)

    return original_results

def draw_keyword_red_box(original_image: np.ndarray, keyword_results: list, output_image_path: str):
    """
    基于原始图片坐标，绘制关键字红框
    :param original_image: 原始OpenCV图片（未缩放）
    :param keyword_results: 原始坐标的关键字结果
    :param output_image_path: 输出图片路径
    """
    try:
        # 校验输入参数是否有效
        if original_image is None or len(original_image.shape) == 0:
            print("❌ 原始图片无效，无法绘制红框")
            return
        if keyword_results is None or len(keyword_results) == 0:
            print("❌ 无关键字结果，无需绘制红框")
            return

        image_with_keyword_box = original_image.copy()
        box_color = (0, 0, 255)
        box_thickness = 2

        # 遍历原始坐标结果，绘制红框
        for keyword_result in keyword_results:
            xmin, ymin, xmax, ymax = keyword_result["keyword_bounding_box"]
            # 防止坐标超出图片范围（超大图片兼容）
            h, w = image_with_keyword_box.shape[:2]
            xmin = max(0, min(xmin, w - 1))
            ymin = max(0, min(ymin, h - 1))
            xmax = max(0, min(xmax, w - 1))
            ymax = max(0, min(ymax, h - 1))

            cv2.rectangle(
                img=image_with_keyword_box,
                pt1=(xmin, ymin),
                pt2=(xmax, ymax),
                color=box_color,
                thickness=box_thickness
            )

        # 保存原始尺寸的带红框图片
        cv2.imwrite(output_image_path, image_with_keyword_box)
        print(f"\n✅ 关键字精准标注图片（原始尺寸）已保存至：{output_image_path}")

    except Exception as e:
        # 输出绘制红框的详细异常
        print(f"\n❌ 关键字红框绘制/保存失败！")
        print(f"异常类型：{type(e).__name__}")
        print(f"异常信息：{str(e)}")
        print(f"详细堆栈（含行号）：")
        traceback.print_exc()

def local_test_text_coords(
    image_path: str,
    target_text: str,
    ignore_case: bool = True
):
    """
    本地测试方法（支持60M超大长截图，纯本地运行）
    """
    try:
        # 1. 验证图片文件是否存在
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在！路径：{image_path}")

        # 2. 验证文件格式
        allowed_extensions = ["jpg", "jpeg", "png"]
        file_ext = image_path.split(".")[-1].lower()
        if file_ext not in allowed_extensions:
            raise ValueError(f"不支持的文件格式！仅支持{allowed_extensions}格式，当前文件：{image_path}")

        # 3. 内存友好型读取超大图片（增加非空校验）
        pil_image = Image.open(image_path)
        if pil_image is None:
            raise Exception("图片读取失败，返回None对象")

        # 转换为RGB格式（避免PNG透明通道问题）
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        image_np = np.array(pil_image)
        # 校验转换后的图片数组是否有效
        if image_np is None or len(image_np.shape) == 0:
            raise Exception("图片转换为NumPy数组失败，返回无效数组")

        # 转换为OpenCV BGR格式
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

        # 4. 获取缩放后关键字结果及缩放因子
        keyword_results_resized, scale_w, scale_h = get_keyword_coordinates(
            image_np=image_np,
            target_text=target_text,
            ignore_case=ignore_case
        )

        # 5. 还原为原始图片坐标
        keyword_results_original = restore_original_coordinates(
            keyword_results=keyword_results_resized,
            scale_w=scale_w,
            scale_h=scale_h
        )

        # 6. 打印结果
        print("=" * 70)
        print(f"本地测试结果汇总（支持超大图片）")
        print(f"原始图片路径：{image_path}")
        print(f"目标关键字：{target_text}")
        print(f"忽略大小写：{ignore_case}")
        print(f"图片缩放因子：{scale_w:.4f}（宽/高）")
        print(f"匹配到关键字数量：{len(keyword_results_original)}")
        print("=" * 70)

        if keyword_results_original:
            for idx, result in enumerate(keyword_results_original, 1):
                print(f"\n第{idx}个关键字匹配结果：")
                print(f"  所在文本块：{result['recognized_text_block']}")
                print(f"  关键字起始位置：{result['keyword_position_in_block']}")
                print(f"  识别置信度：{result['confidence']:.4f}")
                print(f"  关键字精准框（原始坐标）：{result['keyword_bounding_box']}")
                print(f"  原文本块框（原始坐标，参考）：{result['text_block_bounding_box']}")

            # 7. 构造输出图片路径
            image_dir = os.path.dirname(image_path)
            image_name = os.path.basename(image_path).split(f".{file_ext}")[0]
            output_image_name = f"{image_name}_large_image_keyword_box.{file_ext}"
            output_image_path = os.path.join(image_dir, output_image_name)

            # 8. 绘制红框（基于原始图片坐标）
            draw_keyword_red_box(image_np, keyword_results_original, output_image_path)
        else:
            print(f"\n未在图片中找到关键字：{target_text}（不生成标注图片）")

        print("=" * 70)
        return keyword_results_original

    except MemoryError:
        print(f"\n【严重错误】：内存不足！无法处理该超大图片")
        print(f"异常类型：MemoryError")
        print(f"建议：降低 MAX_IMAGE_WIDTH/MAX_IMAGE_HEIGHT 配置，或关闭其他占用内存的程序")
        print(f"详细堆栈（含行号）：")
        traceback.print_exc()
        return []
    except Exception as e:
        print(f"\n【本地测试失败】：")
        print(f"异常类型：{type(e).__name__}")
        print(f"异常信息：{str(e)}")
        print(f"详细堆栈（含行号、调用链）：")
        traceback.print_exc()  # 打印完整堆栈，包含行号
        return []

# 本地测试入口（使用你的参数，纯本地运行，无HTTP相关代码）
if __name__ == "__main__":
    # 你的测试参数（无需修改，支持60M长截图）
    # test_image_path = r'C:\Users\Admin\Desktop\ScreenShot_2025-12-27_163139_119.png'
    # test_image_path = r'input.png'
    # test_target_text = "动态"  # 关键字
    # test_ignore_case = True  # 忽略大小写
    #
    # # 执行超大图片关键字精准定位测试
    # print("🚀 开始执行本地图片关键字定位测试...")
    # local_test_text_coords(
    #     image_path=test_image_path,
    #     target_text=test_target_text,
    #     ignore_case=test_ignore_case
    # )
    # print("\n🏁 本地测试执行完毕！")

    import paddle

    print(paddle.is_compiled_with_cuda())
    print(paddle.device.get_device())

    from paddleocr import PaddleOCR

    ocr = PaddleOCR(use_gpu=True)
    print("OCR GPU OK")
