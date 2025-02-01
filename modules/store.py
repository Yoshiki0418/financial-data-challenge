from glob import glob 
import os
import re
import faiss
import numpy as np
from langchain.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.document_loaders import PyPDFium2Loader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pickle

from .embedded import *

DEPLOYMENT_ID_FOR_EMBEDDING = 'embedding'

class Store:
    # PDFファイルと会社名の対応を設定
    pdf_to_company = {
        "1.pdf": "4℃",
        "2.pdf": "IHI",
        "3.pdf": "NISSAN",
        "4.pdf": "KAGOME",
        "5.pdf": "KITZ",
        "6.pdf": "KUREHA",
        "7.pdf": "GLORY",
        "8.pdf": "サントリー",
        "9.pdf": "ハウス食品",
        "10.pdf": "パナソニック",
        "11.pdf": "Media Do",
        "12.pdf": "MOS",
        "13.pdf": "ライフコーポレーション",
        "14.pdf": "高松コンストラクション",
        "15.pdf": "全国保証株式会社",
        "16.pdf": "東急不動産",
        "17.pdf": "TOYO",
        "18.pdf": "日清食品",
        "19.pdf": "meiji"
    }

    def __init__(
        self, 
        embeddings: Embeddings,
        embedded_path: str,
        faiss_index_path: str,
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        merge_text: bool = True
    ) -> None:
        self.embeddings = embeddings
        self._pdf_files = glob('data/documents/*.pdf')
        self._merge_text = merge_text
        self._documents = []
        self._embedded = []
        self._embedded_path = embedded_path
        self._faiss_index_path = faiss_index_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._create_metadata()
        self._split_docs = self._split_documents(self._documents)

    def _create_metadata(self) -> None:
        """PDFを読み込み、メタデータを作成する"""
        for pdf_path in self._pdf_files:
            file_name = os.path.basename(pdf_path)
            company_name = self.pdf_to_company.get(file_name, "不明な会社")  
            loader = PyPDFium2Loader(pdf_path)
            self._raw_docs: list[Document] = loader.load()

            for doc in self._raw_docs:
                processed_doc = self._process_document(doc, company_name, file_name)
                self._documents.append(processed_doc)

            print(f'{pdf_path} からテキスト抽出終了.')

    def _preprocess_text(self, text: str) -> str:
        """テキスト前処理"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
    
    def _process_document(self, doc: Document, company_name: str, file_name: str) -> Document:
        """PDFから抽出したドキュメントの前処理を行い、Documentオブジェクトを作成"""
        processed_content = self._preprocess_text(doc.page_content)
        return Document(
            page_content=processed_content,
            metadata={
                "company": company_name,
                "source_file": file_name,
                **doc.metadata  
            }
        )
    
    def _split_documents(self, documents) -> list[Document]:
        """テキストを分割"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap
        )
        return text_splitter.split_documents(documents)

    def store_embeddings(self, file_name: str) -> None:
        """埋め込みを作成し、保存"""
        os.makedirs(self._embedded_path, exist_ok=True)

        self._embedded = self.embeddings.embed_documents([doc.page_content for doc in self._split_docs])

        save_path = os.path.join(self._embedded_path, file_name)

        with open(save_path, "wb") as f:
            pickle.dump(self._embedded, f)

        print(f"処理されたドキュメント数: {len(self._split_docs)}")
        print(f"生成された埋め込み数: {len(self._embedded)}")
        print("埋め込みデータを保存しました。")

    def load_embeddings(self, file_name: str) -> None:
        load_path = os.path.join(self._embedded_path, file_name)
        """保存済みの埋め込みデータを読み込む"""
        if os.path.exists(load_path):
            with open(load_path, "rb") as f:
                self._embedded = pickle.load(f)
            print("保存済みの埋め込みデータをロードしました。")
        else:
            print("埋め込みデータが見つかりません。store_embeddings() を実行してください。")

    @property
    def embedded(self) -> list[list[float]]:
        """埋め込みデータを取得"""
        if not self._embedded:
            raise ValueError("埋め込みデータがロードまたは生成されていません。store_embeddings() または load_embeddings() を実行してください。")
        return self._embedded
    
    def old_build_faiss_index(self) -> None:
        """FAISS インデックスを作成し、ローカルに保存"""
        if not self._embedded:
            raise ValueError("埋め込みデータが存在しません。store_embeddings() を実行してください。")
        
        dimension = len(self._embedded[0])  # ベクトルの次元数
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(self._embedded).astype("float32"))

        # Document を ID で管理
        docstore_data = {str(i): doc for i, doc in enumerate(self._split_docs)}
        docstore = InMemoryDocstore(docstore_data)

        index_to_docstore_id = {i: str(i) for i in range(len(self._split_docs))}

        def embedding_function(texts: list[str]) -> list[list[float]]:
            """複数のテキストを Azure OpenAI で埋め込みに変換"""
            return self.embeddings.embed_documents(texts)

        self._vectorstore = FAISS(
            index=faiss_index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
            embedding_function=embedding_function
        )

        os.makedirs("data/processed/old_faiss_index", exist_ok=True)

        self._vectorstore.save_local("data/processed/old_faiss_index")
        print("FAISS インデックスの作成・保存が完了しました！")

    def build_faiss_index(self) -> None:
        """FAISS インデックスを作成し、ローカルに保存"""
        self._vectorstore = FAISS.from_documents(
            documents=self._split_docs,  # メタデータ付きドキュメント
            embedding=self.embeddings  # 事前に作成した埋め込みモデル
        )

        os.makedirs(self._faiss_index_path, exist_ok=True)

        self._vectorstore.save_local(self._faiss_index_path)
        print("FAISS インデックスの作成・保存が完了しました！")

    @property
    def vectorstore(self) -> FAISS:
        return self._vectorstore
        
