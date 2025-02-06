from abc import ABC, abstractmethod
from typing import Any
import pdfplumber


class TableExtractor(ABC):
    """表を抽出する抽象クラス"""

    # フィルタリングの閾値
    MIN_IMAGE_AREA_RATIO = 0.2
    MIN_ASPECT_RATIO = 0.1 
    MAX_TABLE_RATIO = 0.95

    def __init__(self, openai_client: Any, model: str = "4omini") -> None:
        """
        Args:
            openai_client (openai.OpenAI): OpenAI クライアント
            model (str): Azure OpenAI API model name
        """
        self.client = openai_client
        self.model = model

    def filter_tables(self, tables: list, page: pdfplumber.page.Page) -> list:
        """抽出した表をフィルタリングする共通メソッド"""
        valid_tables = []
        page_width, page_height = page.width, page.height

        # **表の座標を `find_tables()` で取得**
        for table_index, table in enumerate(tables):
            if not table or len(table) < 2 or len(table[0]) < 2:
                continue  # 空の表や1行1列しかない表はスキップ

            table_bbox = page.find_tables()  
            if not table_bbox:
                continue 

            x0, y0, x1, y1 = table_bbox[table_index].bbox  # 座標を取得

            # 表の面積とページ全体の割合を計算
            table_width, table_height = x1 - x0, y1 - y0
            table_area_ratio = (table_width * table_height) / (page_width * page_height)
            aspect_ratio = table_width / table_height if table_height > 0 else 0

            # フィルタリング条件
            if (
                table_area_ratio < self.MIN_IMAGE_AREA_RATIO
                or table_area_ratio > self.MAX_TABLE_RATIO
                or aspect_ratio < self.MIN_ASPECT_RATIO
                or aspect_ratio > (1 / self.MIN_ASPECT_RATIO)
            ):
                continue  # 無効な表をスキップ

            valid_tables.append(table)

        return valid_tables

    @abstractmethod
    def extract_tables(self, pdf_path: str) -> dict[int, Any]:
        """
        PDF から表を抽出し、ページ番号をキーとした辞書を返す

        戻り値の形式:
        {
            1: [["A", "B", "C"], ["D", "E", "F"]],  # 1ページ目の表データ
            2: [["X", "Y", "Z"], ["L", "M", "N"]]   # 2ページ目の表データ
        }

        Returns:
            Dict[int, List[List[str]]]: 
                - キー: ページ番号
                - 値: そのページの表（セル値のリスト）
        """
        pass

    def summarize_table(self, table_data: list[list[str]]) -> str:
        """Azure OpenAI APIを用いて表データを要約する"""
        table_text = "\n".join([" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in table_data if row])
        prompt = f"以下の表データを簡潔に要約してください。\n\n{table_text}\n\n要約:"

        system_content = (
            """
            ## 役割

            あなたは抽出された表データを要約する専門家です。以下のプロセスと指示に従って、データの要約を行ってください。

            ## 指示

            1. ユーザーが与えた情報だけをもとに要約してください。\n
            2. 表データに存在しないことは記述しないでください。\n
            3. RAGのコンテキストとして使用されるので、曖昧な情報に変換しないでください。 \n
            4. 表情報を過不足なく要約すること。 \n
            5. 表にある数値はそのまま使用すること。\n
            6. 推測した内容は含まないでください。\n
            7. 表が何を示しているかを推測するのではなく、表の情報を詳細にまとめてください。\n
            8. 各カテゴリに「●」などの記号があり、その記号は何を表しているのかを推測し、数を数えて情報を整理してください。\n

            """
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,  
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=2000
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"⚠️ Azure OpenAI API エラー: {e}")
            return "表データの要約に失敗しました。"


class PdfPlumberExtractor(TableExtractor):
    """pdfplumber を使用して表を抽出"""
    
    def extract_tables(self, pdf_path: str) -> dict:
        tables = {}
        total_extracted = 0
        # NOTE: 抽出する表のフィルタリングする必要がある？ 領域が小さすぎる場合は、無効とする？
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_num = page.page_number
                extracted_tables = page.extract_tables()
                if not extracted_tables:
                    continue  # 表がないページはスキップ

                # **親クラスのフィルタリングメソッドを適用**
                valid_tables = self.filter_tables(extracted_tables, page)

                if valid_tables:
                    tables[page_num] = valid_tables
                    total_extracted += len(valid_tables)
        print(f"✅ 合計 {total_extracted} 個の表を抽出しました。")
        return tables