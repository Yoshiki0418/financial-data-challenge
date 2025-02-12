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
from dotenv import load_dotenv
from typing import Optional, Any
from PyPDF2 import PdfReader

from .embedded import *
from modules.extractors import TableExtractor, PdfPlumberExtractor, AzureExtractor
from modules.extractors.azure_extractor import convert_pdf_page_to_image
from modules.utils import save_extracted_text, extract_images_from_pdf
from modules.extractors.image_extractor import (
    crop_figures_from_image, 
    is_graph_image, 
    get_cropped_image_paths, 
    text_description_image
    )


load_dotenv()

endpoint = os.getenv("AZURE_DOCUMENT_ENDPOINT")
key = os.getenv("AZURE_DOCUMENT_KEY")

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
        client: Any,
        embeddings: Embeddings,
        embedded_path: str,
        faiss_index_path: str,
        extractor_type: str, # 抽出タイプの設定(詳細はconfig)
        chunk_size: int = 2000,
        chunk_overlap: int = 200,
        merge_image_text: bool = True,
        text_extractor_type: str = "pdfplumber", # カスタム抽出時の表抽出機構のタイプ選択
        paipline_config: Optional[dict[str]] = None, # カスタム抽出時のモードを設定
    ) -> None:
        self.embeddings = embeddings
        self._pdf_files = glob('data/documents/*.pdf')
        self._documents = []
        self._embedded = []
        self._embedded_path = embedded_path
        self._faiss_index_path = faiss_index_path
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._images = {}
        self._client = client
        self._merge_image_text = merge_image_text

        if extractor_type == "azure":
            self.azure_extractor = AzureExtractor(endpoint, key)
            self._azure_create_document()
        elif extractor_type == "extracted":
            self._extracted_create_document()
        elif extractor_type == "custom":
            self.table_extractor = self._select_table_instance(text_extractor_type, client)
            print(paipline_config)
            self._create_metadata(paipline_config)
        self._split_docs = self._split_documents(self._documents)


    def _azure_create_document(self) -> None:
        """
        PDFを読み込み、ドキュメントを作成する (汎用化)
        """
        for pdf_path in self._pdf_files:
            file_name = os.path.basename(pdf_path)
            company_name = self.pdf_to_company.get(file_name, "不明な会社")  

            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"{file_name} は {num_pages} ページあります。")

            for i in range(num_pages):
                page_number = i + 1
                temp_image_path = convert_pdf_page_to_image(pdf_path, i + 1)
                
                # Azure の抽出機構を利用してページのテキストを抽出
                extraction_result = self.azure_extractor.extract_text_from_local(temp_image_path)
                markdown_text = extraction_result.get("markdown", "")

                # コンテキストを保存する
                save_extracted_text(pdf_path, page_number, markdown_text)

                # 座標位置に基づいて図や画像のクロッピングを行う
                cropped_files = crop_figures_from_image(temp_image_path, extraction_result["figure_positions"])

                # クロップした画像でグラフ以外を排除
                graph_files = is_graph_image(cropped_files) if cropped_files else []

                # グラフ画像をLLMでテキスト化
                image_texts = []
                if graph_files:
                    for graph_file in graph_files:
                        image_text = text_description_image(graph_file, self._client)
                        image_texts.append(image_text)

                    # テキストを結合
                    merged_image_text = "\n\n".join(image_texts)
                    save_extracted_text(pdf_path, page_number, merged_image_text, sub_dir="image_text")

                # ページ番号などのメタデータを付与して Document オブジェクトを作成
                metadata = {
                    "page": i + 1,
                    "source_file": file_name,
                    "image_path": temp_image_path,
                    "company": company_name,
                    "grapf_files": graph_files,
                }
                doc = Document(page_content=markdown_text, metadata=metadata)
                self._documents.append(doc)

                # 画像説明テキストの統合方法をユーザー設定で変更
                if self._merge_image_text:
                    # 画像の説明テキストをマークダウンに結合
                    merged_text = markdown_text + "\n\n" + "\n".join(image_texts) if image_texts else markdown_text
                    doc = Document(page_content=merged_text, metadata=metadata)
                    self._documents.append(doc)
                else:
                    # 画像の説明テキストを別のチャンクとして格納
                    doc_main = Document(page_content=markdown_text, metadata=metadata)
                    self._documents.append(doc_main)

                    for image_text, graph_file in zip(image_texts, graph_files):
                        image_metadata = {
                            "page": i + 1,
                            "source_file": file_name,
                            "company": company_name,
                            "image_path": graph_file, 
                        }
                        doc_image = Document(page_content=image_text, metadata=image_metadata)
                        self._documents.append(doc_image)
    

    def _extracted_create_document(
            self,
            use_image: bool = True, 
            extracted_image_text: bool = True,
    )-> None:
        """
        PDFごとに抽出済みのMarkdownテキストファイルからテキストを読み込み、
        Documentオブジェクトを作成して self._documents に追加する。
        """
        for pdf_path in self._pdf_files:
            file_name = os.path.basename(pdf_path)
            base_name = os.path.splitext(file_name)[0]
            company_name = self.pdf_to_company.get(file_name, "不明な会社")  
            graph_files = []

            reader = PdfReader(pdf_path)
            num_pages = len(reader.pages)
            print(f"{file_name} は {num_pages} ページあります。")

            # PDFごとの抽出済みテキストが格納されているディレクトリパスを作成
            base_dir = os.path.join("data", "extractored_txt", base_name)

            for i in range(num_pages):
                page_number = i + 1

                # テキストファイル名は "page_<ページ番号>.txt" となっているとする
                read_text_path = os.path.join(base_dir, f"page_{page_number}.txt")
                if not os.path.exists(read_text_path):
                    raise FileNotFoundError(f"[ERROR] {read_text_path} が存在しません。")
                with open(read_text_path, "r", encoding="utf-8") as f:
                    markdown_text = f.read()  
                
                graph_files = []
                image_texts = []

                # 画像データを埋め込むか
                if use_image:
                    if extracted_image_text:
                        image_base_dir = os.path.join("data", "image_text", base_name)
                        read_image_text_path = os.path.join(image_base_dir, f"page_{page_number}.txt")
                        if os.path.exists(read_image_text_path): 
                            with open(read_image_text_path, "r", encoding="utf-8") as f:
                                existing_image_text = f.read().strip()
                                if existing_image_text:
                                    image_texts.append(existing_image_text)
                        else:
                            pass
                    else:
                        # クロップ済みの画像ファイルを抽出する
                        cropped_files = get_cropped_image_paths(base_name, page_number)

                        # クロップした画像でグラフ以外を排除
                        if cropped_files:
                            graph_files = is_graph_image(cropped_files)

                            # LLM で画像を説明する
                            if graph_files:
                                for graph_file in graph_files:
                                    image_text = text_description_image(graph_file, self._client)
                                    image_texts.append(image_text)
                                # テキストを結合
                                merged_image_text = "\n\n".join(image_texts)
                                save_extracted_text(pdf_path, page_number, merged_image_text, sub_dir="image_text")
                        else:
                            print("クロップされた画像がありません。")    

                # メタデータの作成
                metadata = {
                    "page": page_number,
                    "source_file": file_name,
                    "company": company_name,
                    "graph_files": graph_files,
                }

                # 画像説明テキストの統合方法をユーザー設定で変更
                if self._merge_image_text:
                    # 画像の説明テキストをマークダウンに結合
                    merged_text = markdown_text + "\n\n" + "\n".join(image_texts) if image_texts else markdown_text
                    doc = Document(page_content=merged_text, metadata=metadata)
                    self._documents.append(doc)
                else:
                    # 画像の説明テキストを別のチャンクとして格納
                    doc_main = Document(page_content=markdown_text, metadata=metadata)
                    self._documents.append(doc_main)

                    for image_text, graph_file in zip(image_texts, graph_files):
                        image_metadata = {
                            "page": page_number,
                            "source_file": file_name,
                            "company": company_name,
                            "image_path": graph_file, 
                        }
                        doc_image = Document(page_content=image_text, metadata=image_metadata)
                        self._documents.append(doc_image)


    def _create_metadata(self, config: dict = None) -> None:
        """
        PDFを読み込み、メタデータを作成する (汎用化)
        
        Args:
            config (dict): 処理のON/OFFを設定する辞書
                - "include_tables": 表をテキストに結合するか (True/False)
                - "linearize_tables": 表の線形化をメタデータに保存するか (True/False)
                - "summarize_tables": LLMで表を要約して結合するか (True/False)
                - "caption_images": 画像のキャプションを生成しテキストに結合するか (True/False)
                - "store_image_captions": 画像キャプションをメタデータに保存するか (True/False)
        """
        if config is None:
            config = {
                "include_tables": True,
                "linearize_tables": False,
                "summarize_tables": False,
                "caption_images": False,
                "store_image_captions": True
            }

        for pdf_path in self._pdf_files:
            file_name = os.path.basename(pdf_path)
            company_name = self.pdf_to_company.get(file_name, "不明な会社")  

            # **テキストの取得**
            loader = PyPDFium2Loader(pdf_path)
            raw_docs = loader.load()

            # **表の取得**
            tables = self.table_extractor.extract_tables(pdf_path)

            # # **画像キャプションの取得 (ページ単位)**
            # image_paths = extract_images_from_pdf(pdf_path, file_name)
            # images = self._process_image(image_paths)
            images = None

            for doc in raw_docs:
                processed_doc = self._process_document(doc, company_name, file_name, config, tables, images)
                self._documents.append(processed_doc)

            print(f'{pdf_path} からテキスト抽出終了.')


    def _preprocess_text(self, text: str) -> str:
        """テキスト前処理"""
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'[^\w\s]', '', text)
        text = text.lower()
        return text
    

    def _process_document(self, doc: Document, company_name: str, file_name: str, config: dict, tables: dict, images: dict) -> Document:
        """PDFから抽出したドキュメントの前処理を行い、Documentオブジェクトを作成"""
        page_num = doc.metadata.get("page")
        processed_content = self._preprocess_text(doc.page_content)

        # **該当ページの表を結合**
        if config["include_tables"] and page_num in tables:
            table_list = tables[(page_num)]  # ページ内のすべての表
            print(f"📄 処理中: {file_name} | ページ番号: {page_num}")

            table_summaries = []  # 各表ごとの要約リスト

            for idx, table_data in enumerate(table_list):
                if config["summarize_tables"]:
                    table_summary = self.table_extractor.summarize_table(table_data)
                    print(table_summary)
                    print("")
                    table_summaries.append(table_summary)
                else:
                    formatted_table = "\n".join([" | ".join([str(cell) if cell is not None else "" for cell in row]) for row in table_data if row])
                    table_summaries.append(formatted_table)

            processed_content += "\n\n" + "\n\n".join(table_summaries)

        # **メタデータの作成**
        metadata = {
            "company": company_name,
            "source_file": file_name,
            **doc.metadata
        }

        # **表をメタデータに保存**
        if config["linearize_tables"] and (file_name, page_num) in tables:
            metadata["linearized_table"] = tables[(file_name, page_num)]

        # **画像キャプションの追加**
        if config["store_image_captions"] and (file_name, page_num) in images:
            metadata["image_captions"] = images[(file_name, page_num)]

        return Document(page_content=processed_content, metadata=metadata)
        
    
    def _split_documents(self, documents) -> list[Document]:
        """テキストを分割"""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self._chunk_size,
            chunk_overlap=self._chunk_overlap
        )
        return text_splitter.split_documents(documents)
    

    def _select_table_instance(self, text_extractor_type: str, client: Any) -> TableExtractor:
        match text_extractor_type.lower():
            case "pdfplumber":
                return PdfPlumberExtractor(client)
            # case "camelot":
            #     return CamelotExtractor()
            # case "pdf2docx":
            #     return Pdf2DocxExtractor()
            case _:
                raise ValueError(f"未知の表抽出方法: {text_extractor_type}")


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
        
