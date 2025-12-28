# -*- coding: utf-8 -*-
# time: 2025/12/27 21:18
# file: ocr_runner.py
# author: RPA高老师
import cv2
from paddleocr import PaddleOCR
from typing import List, Dict


class OCRRunner:
    def __init__(
        self,
        use_gpu: bool = True,
        lang: str = "ch",
        det_limit_side_len: int = 960,
    ):
        self.ocr = PaddleOCR(
            use_gpu=use_gpu,
            lang=lang,
            det=True,
            rec=True,
            cls=False,
            det_limit_side_len=det_limit_side_len,
            show_log=False,
        )

    def locate_keyword(
        self,
        image_path: str,
        keyword: str,
        draw: bool = True,
        output_path: str = "result.png",
    ) -> List[Dict]:

        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(image_path)

        ocr_result = self.ocr.ocr(image_path, cls=False)

        hits = []
        seen_boxes = set()

        for line in ocr_result[0]:
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

if __name__ == "__main__":
    runner = OCRRunner(use_gpu=True)

    res = runner.locate_keyword(
        image_path=r"C:\Users\Admin\Desktop\2025-12-27_095600.png",
        keyword="品质",
        draw=True,
        output_path="./output/result.png",
    )

    print(f"命中数量：{len(res)}")
    for r in res:
        print(r)
