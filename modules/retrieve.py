import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import Any
import re

from .embedded import *

class Retrieve:
    company_metadata = {
        "4℃": ["4℃", "ヨンドシー", "ヨンドシージュエリー"],
        "IHI": ["IHI", "石川島播磨", "石川島播磨重工業"],
        "NISSAN": ["NISSAN", "日産", "日産自動車"],
        "KAGOME": ["KAGOME", "カゴメ"],
        "KITZ": ["KITZ", "キッツ"],
        "KUREHA": ["KUREHA", "クレハ", "株式会社クレハ"],
        "GLORY": ["GLORY", "グロリー", "グローリー"],
        "サントリー": ["サントリー", "Suntory"],
        "ハウス食品": ["ハウス食品", "House Foods"],
        "パナソニック": ["パナソニック", "Panasonic", "パナソニックグループ"],
        "Media Do": ["Media Do", "メディアドゥ"],
        "MOS": ["MOS", "モスフードサービス", "モスバーガー", "モスグループ"],
        "ライフコーポレーション": ["ライフコーポレーション", "ライフ"],
        "高松コンストラクション": ["高松コンストラクション", "高松建設"],
        "全国保証株式会社": ["全国保証株式会社", "全国保証"],
        "東急不動産": ["東急不動産", "Tokyu Real Estate", "Tokyu"],
        "TOYO": ["TOYO", "東洋ゴム", "TOYO TIRES", "東洋エンジニアリング"],
        "日清食品": ["日清食品", "Nissin", "Nissin Foods"],
        "meiji": ["meiji", "明治", "明治製菓", "Meiji Seika"]
    }

    def __init__(self, embeddings: Embeddings, faiss_index_path: str) -> None:
        print("start Retrieve")
        self.embeddings = embeddings
        self.vectorstore = FAISS.load_local(
            folder_path=faiss_index_path,
            embeddings=self.embeddings,
            allow_dangerous_deserialization=True
        )
        self.validate_index()

    def validate_index(self):
        """FAISS インデックスの整合性をチェック"""
        index_size = self.vectorstore.index.ntotal
        docstore_size = len(self.vectorstore.index_to_docstore_id)

        if index_size != docstore_size:
            print(f"[警告] FAISS インデックスと `index_to_docstore_id` のサイズが一致しません。")
            print(f"FAISS インデックスサイズ: {index_size}")
            print(f"index_to_docstore_id のサイズ: {docstore_size}")

    def extract_company_name(self, query: str) -> str:
        """質問から企業名を抽出する"""
        for company, aliases in self.company_metadata.items():
            for alias in aliases:
                # 🔹 `\b` を削除し、部分一致するように修正
                if re.search(re.escape(alias), query, re.IGNORECASE):
                    return company  # 一致した企業名を返す
        return None

    # def search(self, query: str, top_k: int = 5) -> list[Document]:
    #     """クエリをベクトル化し、FAISSインデックスから類似検索"""
    #     return self.vectorstore.similarity_search(query, k=top_k)

    def old_search(self, query: str, top_k: int = 5, return_text: bool = True) -> list[Document] | str:
        """クエリをベクトル化し、FAISSインデックスから類似検索を行う
    
        Args:
            query (str): 検索クエリ文字列
            top_k (int, optional): 取得する類似文書の最大数. Defaults to 5.
            return_text (bool, optional): Trueの場合、検索結果を結合した文字列で返す. Defaults to False.

        Returns:
            Union[list[Document], str]: return_text=True の場合は文字列、False の場合は Document のリスト
        """
        query_embedding = self.embeddings.embed_query(query)
        relevant_docs = self.vectorstore.similarity_search_by_vector(
            np.array(query_embedding, dtype=np.float32), k=top_k
        )
        if return_text:
            return "\n\n".join([doc.page_content for doc in relevant_docs])
        return relevant_docs
    
    def search(self, query: str, top_k: int = 5, fetch_k: int = 20, return_text: bool = True) -> list[Document] | str:
        """クエリをベクトル化し、FAISSインデックスから類似検索を行う
        - 質問に企業名がある場合に、付属しているメタ情報の企業名と一致しているドキュメントだけをフィルタリング
        した上で、類似度検索を行う。
    
        Args:
            query (str): 検索クエリ文字列
            top_k (int, optional): 取得する類似文書の最大数. Defaults to 5.
            return_text (bool, optional): Trueの場合、検索結果を結合した文字列で返す. Defaults to False.

        Returns:
            Union[list[Document], str]: return_text=True の場合は文字列、False の場合は Document のリスト
        """
        # ** 企業名の抽出 **
        company_name = self.extract_company_name(query)

        query_embedding = self.embeddings.embed_query(query)

        if company_name:
            print("")
            print(f"企業名 '{company_name}' を検出。対象企業のドキュメントのみを検索します。")

            # 企業名でフィルタリングするため、メタデータを利用
            relevant_docs = self.vectorstore.similarity_search_by_vector(
                np.array(query_embedding, dtype=np.float32), 
                k=top_k, 
                filter={"company": company_name},  # 企業名をフィルタリング
                fetch_k=fetch_k
            )
            print(f"フィルタ後のドキュメント数: {len(relevant_docs)} 件")
            print("")
        else:
            print("")
            print("企業名が特定できませんでした。全ドキュメントから検索します。")
            print("")
            relevant_docs = self.vectorstore.similarity_search_by_vector(
                np.array(query_embedding, dtype=np.float32), k=top_k
            )
    
        if return_text:
            return "\n\n".join([doc.page_content for doc in relevant_docs])
        return relevant_docs
