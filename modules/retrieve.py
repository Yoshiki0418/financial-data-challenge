import numpy as np
from langchain.vectorstores import FAISS
from langchain.schema import Document
from typing import Any

from .embedded import *

class Retrieve:
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

    # def search(self, query: str, top_k: int = 5) -> list[Document]:
    #     """クエリをベクトル化し、FAISSインデックスから類似検索"""
    #     return self.vectorstore.similarity_search(query, k=top_k)
    
    def search(self, query: str, top_k: int = 5, return_text: bool = True) -> list[Document] | str:
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
