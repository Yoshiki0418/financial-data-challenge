import re
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest, DocumentContentFormat
from pdf2image import convert_from_path
import os
from azure.core.exceptions import HttpResponseError


def convert_pdf_page_to_image(pdf_path: str, page_number: int, dpi: int = 300) -> str:
        """
        pdf2image を利用して、指定されたPDFの1ページを画像に変換し、一時ファイルとして保存してパスを返す。
        dpi は必要に応じて調整してください。
        
        :param pdf_path: PDFファイルのパス
        :param page_number: 変換するページ番号 (1始まり)
        :param dpi: 解像度（デフォルト150）
        :return: 変換後の画像ファイルパス（例: temp_<元ファイル名>_page_<ページ番号>.png）
        """
        # convert_from_path はページ番号の指定が1始まり
        images = convert_from_path(pdf_path, dpi=dpi, first_page=page_number, last_page=page_number)
        if not images:
            raise ValueError(f"ページ {page_number} の画像変換に失敗しました。")
        image = images[0]
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        temp_image_path = f"data/pdf_image/{base_name}_page_{page_number}.png"
        image.save(temp_image_path, "PNG")
        return temp_image_path

def convert_html_table_to_markdown(html_table: str) -> str:
    """
    HTML の <table> ブロックを Markdown テーブルに変換する関数です。
    <caption> タグがある場合はその内容を抽出し、テーブルの上部に追加します。
    各行は <tr>～</tr> で抽出し、セルは <th> または <td> 内の内容を取得します。
    タグ内の属性にも対応するため、正規表現で [^>]* を利用しています。
    """
    import re

    # <caption> の内容を抽出（存在する場合）
    caption_match = re.search(r"<caption[^>]*>(.*?)</caption>", html_table, flags=re.DOTALL)
    caption_text = caption_match.group(1).strip() if caption_match else ""
    
    # <caption> タグを削除してから行抽出を行う
    html_table_no_caption = re.sub(r"<caption[^>]*>.*?</caption>", "", html_table, flags=re.DOTALL)
    
    # 各行 (<tr>～</tr>) を抽出
    rows = re.findall(r"<tr>(.*?)</tr>", html_table_no_caption, flags=re.DOTALL)
    table_data = []
    for row in rows:
        # セル抽出：<th> または <td> タグ内のテキスト（属性にも対応）
        if "<th" in row:
            cells = re.findall(r"<th[^>]*>(.*?)</th>", row, flags=re.DOTALL)
        else:
            cells = re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.DOTALL)
        # 各セル内の余分な空白や改行を統一
        cells = [re.sub(r"\s+", " ", cell.strip()) for cell in cells]
        table_data.append(cells)
    
    # テーブルデータが取得できなかった場合、caption があればそれだけ返す
    if not table_data:
        return f"**{caption_text}**" if caption_text else ""
    
    # 最初の行をヘッダー行とする
    header = table_data[0]
    data_rows = table_data[1:] if len(table_data) > 1 else []
    
    md_table = ""
    # caption が存在すればテーブルの上に追加（例として太字表示）
    if caption_text:
        md_table += f"**{caption_text}**\n\n"
    
    # Markdown テーブル形式に変換
    md_table += "| " + " | ".join(header) + " |\n"
    md_table += "| " + " | ".join(["---"] * len(header)) + " |\n"
    for row in data_rows:
        md_table += "| " + " | ".join(row) + " |\n"
    return md_table



class AzureExtractor:
    def __init__(self, endpoint: str, key: str):
        """
        Initialize the Document Intelligence Client.
        :param endpoint: Azure Document Intelligence endpoint.
        :param key: Azure Document Intelligence API key.
        """
        self.client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))

    def _compress_image(self, image_path: str, quality: int = 80, dpi: int = 150):
        """
        画像を開き、DPI を指定しつつ JPEG 形式で上書き保存します。
        ピクセル数によるリサイズは行いません。

        :param image_path: 画像ファイルパス
        :param quality: JPEG で保存する際の品質（0～100）
        :param dpi: 変更後の DPI (例:150)
        """
        from PIL import Image

        with Image.open(image_path) as img:
            # 画像を RGB に変換
            img = img.convert("RGB")
            # JPEG 形式に変換し、dpi を指定
            img.save(
                image_path,
                "JPEG",
                quality=quality,
                dpi=(dpi, dpi)
            )
            print(f"[INFO] 画像をJPEG化 + DPI変更: {image_path}, quality={quality}, size={img.size}, dpi={dpi}")

    def extract_text_from_local(self, image_path: str):
        """
        ローカルの PDF ファイルからテキスト、テーブル、図の位置情報を抽出し、Markdown 形式で返す。
        出力処理には output_content_format=DocumentContentFormat.MARKDOWN を利用します。
        
        - <figure> ブロックは、内部に <figcaption> があってもその内容を判定対象から除外し、
          14文字以上の行が存在すれば文章形式とみなしてタグのみ除去、そうでなければ全体を削除します。
        - <table> ブロックは、パイプ区切りの Markdown テーブルに変換します。
        - ページ情報（<!-- PageHeader="..." --> 等）は削除します。
        - 連続する空白行は 2 行に正規化します。
        
        :param pdf_path: ローカルの PDF ファイルのパス
        :return: {'markdown': 抽出結果の Markdown テキスト, 'figure_positions': 図の位置情報のリスト}
        """
        max_retries = 1  # 1度リトライしてダメなら諦める
        for attempt in range(max_retries + 1):
            try:
                with open(image_path, "rb") as f:
                    poller = self.client.begin_analyze_document(
                        model_id="prebuilt-layout",
                        body=f,
                        output_content_format=DocumentContentFormat.MARKDOWN
                    )
                    result = poller.result()
                # ここまでで成功すれば break
                break
            except HttpResponseError as e:
                if "InvalidContentLength" in str(e) and attempt < max_retries:
                    # 初回エラー時 -> 画質を下げて再試行
                    print("[WARN] 画像サイズが大きいためエラー発生。画質を下げて再試行します。")
                    self._compress_image(image_path, quality=70, dpi=150)
                else:
                    # 2回目も失敗、または別エラーの場合は空データを返す
                    print("[ERROR] 解析に失敗。空の結果を返します。")
                    return {
                        "markdown": "",
                        "figure_positions": []
                    }

        # Markdown 形式の出力は result.content に格納される
        markdown_text = result.content

        # --- ページ情報の除去 ---
        # <!-- PageHeader="..." -->、<!-- PageFooter="..." -->、<!-- PageNumber="..." --> を削除
        markdown_text = re.sub(r"<!--\s*(PageHeader|PageFooter|PageNumber)=\".*?\"\s*-->\n?", "", markdown_text)

        # --- <figure> ブロックの処理 ---
        # ヒューリスティック: ブロック内の各行のうち、<figcaption> の内容は除去し、
        # 14文字以上の行があれば文章形式とみなす
        def is_descriptive(text):
            # <figcaption> ... </figcaption> の内容を除去
            text_without_figcaption = re.sub(r'<figcaption>.*?</figcaption>', '', text, flags=re.DOTALL)
            lines = [line.strip() for line in text_without_figcaption.splitlines() if line.strip()]
            for line in lines:
                if len(line) >= 9:
                    return True
            return False

        def replace_figure_block(match):
            content = match.group(1)
            figcaption_match = re.search(r'(<figcaption[^>]*>.*?</figcaption>)', content, flags=re.DOTALL)
            if figcaption_match:
                return figcaption_match.group(1).strip()
            # <figcaption> が無い場合は、従来の処理を行う
            if is_descriptive(content):
                return content.strip()
            else:
                return ""

        markdown_text = re.sub(r"<figure>(.*?)</figure>", replace_figure_block, markdown_text, flags=re.DOTALL).strip()

        # --- <table> ブロックの処理 ---
        # HTML の <table> ブロックをパイプ区切りの Markdown テーブルに変換
        def replace_table_block(match):
            html_table = match.group(0)
            return convert_html_table_to_markdown(html_table)
        
        markdown_text = re.sub(r"<table>.*?</table>", replace_table_block, markdown_text, flags=re.DOTALL).strip()

        # --- 空白行の正規化 ---
        # 連続する空白行が2行以上ある場合、2行にまとめる
        markdown_text = re.sub(r'\n{3,}', "\n\n", markdown_text)

        # --- ※のすぐ後ろに数字がある場合、その※と数字を除去 ---
        # 例: "*3" のようなパターンを削除（※として "*" を使用）
        markdown_text = re.sub(r'※3(?=\d)', '', markdown_text)

        # **図領域をまとめる**
        figure_positions = []
        if getattr(result, "figures", None):
            for i, fig in enumerate(result.figures):
                print(i)
                print(fig)
                if not getattr(fig, "bounding_regions", []):
                    # バウンディング領域自体が無いならスキップ
                    continue

                # bounding_regions が複数の場合もありうるのでループ
                for region in fig.bounding_regions:
                    polygon_coords = list(zip(region.polygon[::2], region.polygon[1::2]))
                    
                    figure_positions.append({
                        "figure_index": i + 1,         
                        "page_number": region.page_number,
                        "polygon": polygon_coords,
                    })

        return {
            "markdown": markdown_text,
            "figure_positions": figure_positions
        }


# if __name__ == "__main__":
#     endpoint = "https://dcoumentai2020.cognitiveservices.azure.com/"
#     key = "A6vCfGQPoxbOa9UpTHmVlPODoYEowyFM60SCrdD2amqcsNZpff7bJQQJ99BBACYeBjFXJ3w3AAALACOGGjbN"
#     img_path = "../../data/pdf_image/13_page_34.png" 

#     extractor = AzureExtractor(endpoint, key)

#     result = extractor.extract_text_from_local(img_path)
#     print(result["markdown"])
#     print(result["figure_positions"])
#     output_dir = "../../data/cropped_figures"
#     cropped_files = crop_figures_from_image(img_path, result["figure_positions"], output_dir)


