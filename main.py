import hydra
import pandas as pd
from omegaconf import DictConfig
from openai import AzureOpenAI, OpenAI
import os
from dotenv import load_dotenv
import logging

from modules.get_query import get_query
from modules import Store, Retrieve, AugmentPrompt, Generate, get_embedded, create_embedding_instance

load_dotenv()

# httpx のログレベルを WARNING 以上に設定（INFO を抑制）
logging.getLogger("httpx").setLevel(logging.WARNING)

# 環境変数から設定を取得
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT_ID_FOR_EMBEDDING = os.getenv("DEPLOYMENT_ID_FOR_EMBEDDING")

# OpenAIのAPIキー
OPENAI_API_KEY = os.getenv("OPEN_API_KEY")
DEPLOYMENT_ID_FOR_CHAT_COMPLETION = os.getenv("DEPLOYMENT_ID_FOR_CHAT_COMPLETION")

# Azureの埋め込み用クライアント
azure_client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION
)

# OpenAIの質問生成用クライアント
openai_clinet = OpenAI(
    api_key=OPENAI_API_KEY
)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    # 埋め込みタイプの選択
    StoreEmbeddingClass = get_embedded(args.embedded_type.store)
    RetrieveEmbeddingClass = get_embedded(args.embedded_type.retrieve)

    # ストアと埋め込みクラスの初期化
    store_embeddings = create_embedding_instance(StoreEmbeddingClass, client=azure_client, model_name=None)
    retrieve_embeddings = create_embedding_instance(RetrieveEmbeddingClass, client=azure_client, model_name=None)

    # FAISSをビルド
    if args.create_faiss:
        store = Store(client=azure_client, embeddings=store_embeddings, **args.store)
        # store.build_faiss_index()

    # # RAGの初期化
    # retrieve = Retrieve(retrieve_embeddings, args.store.faiss_index_path)
    # augment = AugmentPrompt(**args.augmentprompt)
    # generate = Generate(client=openai_clinet, **args.generate)

    # # 質問を受け取る
    # queries: list[str] = get_query()
    # results = []
    # output_dir = "data/logs"
    # os.makedirs(output_dir, exist_ok=True)
    
    # for idx, query in enumerate(queries):
    #     print("-------------------------------------------------------------------------------------------")
    #     print(f"質問{idx}: {query}")

    #     # 質問を入力して検索
    #     relevant_docs = retrieve.search(query=query, **args.retrieve.search)

    #     # プロンプトの構築
    #     messages = augment.build_prompt(query, relevant_docs)  

    #     # NOTE: 類似度で引っかかったコンテキストのメタデータから、pdfとページを受け取り、そのページを画像に変換してVQA形式で解かせる。

    #     # 回答生成
    #     answer = generate.generate_answer(messages)
    #     results.append([idx, answer])

    #     # **出力ファイルに書き込む**
    #     output_file = os.path.join(output_dir, f"query_{idx}.txt")
    #     with open(output_file, "w", encoding="utf-8") as f:
    #         f.write(f"質問: {query}\n\n")
    #         f.write("=== コンテキスト ===\n")
    #         f.write(f"{relevant_docs}\n")
    #         f.write("=== 回答 ===\n")
    #         f.write(answer + "\n")
    
    #     print("")

    # # csvファイルを構築する
    # results_df = pd.DataFrame(results)
    # results_df.to_csv('data/sample_submit/predictions.csv', index=False, header=False)


if __name__ == "__main__":
    run()