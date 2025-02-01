from typing import Any
from abc import ABC, abstractmethod
from langchain.embeddings.base import Embeddings
from langchain.embeddings.base import Embeddings
from concurrent.futures import ThreadPoolExecutor
# from transformers import AutoTokenizer, AutoModel

DEPLOYMENT_ID_FOR_EMBEDDING = 'embedding'

class BaseEmbeddings(Embeddings, ABC):
    """埋め込みモデルの抽象クラス"""

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """複数ドキュメントの埋め込みを生成"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        """単一のクエリを埋め込み"""
        pass


class AzureOpenAIEmbeddings(BaseEmbeddings):
    """LangChainのEmbeddings抽象クラスを継承して、AzureOpenAIの埋め込みを行う"""
    def __init__(self, client: Any) -> None:
            self.client = client

    def embed_documents(self, texts: list[str]) -> list[list[float]]: 
        """複数のドキュメントを並列で埋め込み"""
        with ThreadPoolExecutor(max_workers=4) as executor:
            embeddings = list(executor.map(self._get_embedding, texts))
        return embeddings

    def embed_query(self, text: str) -> list[float]:
        """単一の検索クエリを埋め込み"""
        return self._get_embedding(text)

    def _get_embedding(self, text: str) -> list[float]:
        """Azure OpenAI でテキストの埋め込みを取得"""
        response = self.client.embeddings.create(
            input=text,
            model=DEPLOYMENT_ID_FOR_EMBEDDING  # 適切なモデルIDを指定
        )
        return response.data[0].embedding
        

# class BERTEmbeddings(BaseEmbeddings):
#     """BERT を使った埋め込みクラス"""

#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> None:
#         """モデルの初期化"""
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = AutoModel.from_pretrained(model_name)

#     def embed_documents(self, texts: list[str]) -> list[list[float]]:
#         """複数ドキュメントを埋め込み"""
#         return [self.get_embedding(t) for t in texts]

#     def embed_query(self, text: str) -> list[float]:
#         """単一の検索クエリを埋め込み"""
#         return self.get_embedding(text)

#     def get_embedding(self, text: str) -> list[float]:
#         """BERT を使って埋め込みを取得"""
#         inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)