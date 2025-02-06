# utils.py
import os


def save_extracted_text(pdf_path: str, page_number: int, text_content: str):
    """
    PDFのページごとの抽出テキストをファイルとして保存します。
    
    - data/extractored_txt/{PDFファイル名}/page_{ページ番号}.txt という構成で保存します。
    - 例: data/extractored_txt/9/page_1.txt

    Args:
        pdf_path: 抽出元PDFファイルのパス
        page_number: ページ番号 (1始まりなど)
        text_content: 抽出したテキスト
    """
    # PDFのファイル名(拡張子除く)を取得
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]

    # PDFごとのディレクトリを作成
    output_dir = os.path.join("data", "extractored_txt2", base_name)
    os.makedirs(output_dir, exist_ok=True)

    # ページごとのテキストファイル名
    output_path = os.path.join(output_dir, f"page_{page_number}.txt")

    # テキストファイルとして書き込み
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text_content)


MIN_IMAGE_AREA_RATIO = 0.2  # ページ全体の 20% 未満なら無視
MIN_ASPECT_RATIO = 0.1  # 極端に細長い画像を除外

def extract_images_from_pdf(pdf_path: str, file_name: str) -> list[str]:
        """PDF 内の画像・表を抽出し、不要なものをフィルタリング"""
        output_dir = "data/images"
        os.makedirs(output_dir, exist_ok=True)

        extracted_images = []

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)

            for page_num in range(total_pages):
                page = pdf.pages[page_num]
                image_objects = page.images
                tables = page.find_tables()

                if not image_objects and not tables:
                    continue  # 画像・表がないページはスキップ

                # **1ページずつ処理してメモリを抑える**
                pdf_image = convert_from_path(pdf_path, dpi=300, first_page=page_num+1, last_page=page_num+1)[0]
                img_width, img_height = pdf_image.size
                pdf_width, pdf_height = page.width, page.height

                def process_crop_and_save(x0, y0, x1, y1, prefix, index):
                    """クロップ処理 & フィルタリング"""
                    x0, y0 = max(0, int((x0 / pdf_width) * img_width)), max(0, int((y0 / pdf_height) * img_height))
                    x1, y1 = min(img_width, int((x1 / pdf_width) * img_width)), min(img_height, int((y1 / pdf_height) * img_height))

                    width, height = x1 - x0, y1 - y0
                    area_ratio = (width * height) / (img_width * img_height)
                    aspect_ratio = width / height if height > 0 else 0

                    # **フィルタリング条件**
                    if (
                        area_ratio < MIN_IMAGE_AREA_RATIO  # 面積が小さすぎる
                        or aspect_ratio < MIN_ASPECT_RATIO  # 極端に細長い
                        or aspect_ratio > (1 / MIN_ASPECT_RATIO)  # 極端に横長
                    ):
                        return  # 無効な領域ならスキップ

                    cropped_image = pdf_image.crop((x0, y0, x1, y1))
                    image_path = os.path.join(output_dir, f"{file_name}_page_{page_num+1}_{prefix}_{index+1}.png")
                    cropped_image.save(image_path, "PNG")
                    extracted_images.append(image_path)

                # **画像の処理**
                for img_index, img in enumerate(image_objects):
                    process_crop_and_save(img["x0"], img["top"], img["x1"], img["bottom"], "img", img_index)

                # **表の処理**
                for table_index, table in enumerate(tables):
                    process_crop_and_save(table.bbox[0], table.bbox[1], table.bbox[2], table.bbox[3], "table", table_index)

                del pdf_image
                gc.collect()

        print(f"{file_name}: {len(extracted_images)} 枚の画像・表を抽出しました。")
        return extracted_images