import hydra
import pandas as pd
from omegaconf import DictConfig
from openai import AzureOpenAI
import os
from dotenv import load_dotenv
import logging

from modules.get_query import get_query
from modules import Store, Retrieve, AugmentPrompt, Generate, get_embedded

load_dotenv()

# httpx のログレベルを WARNING 以上に設定（INFO を抑制）
logging.getLogger("httpx").setLevel(logging.WARNING)

# 環境変数から設定を取得
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_VERSION = os.getenv("API_VERSION")
DEPLOYMENT_ID_FOR_CHAT_COMPLETION = os.getenv("DEPLOYMENT_ID_FOR_CHAT_COMPLETION")
DEPLOYMENT_ID_FOR_EMBEDDING = os.getenv("DEPLOYMENT_ID_FOR_EMBEDDING")

client = AzureOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=API_VERSION
)

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    # 埋め込みタイプの選択
    StoreEmbeddingClass = get_embedded(args.embedded_type.store)
    RetrieveEmbeddingClass = get_embedded(args.embedded_type.retrieve)

    # ストアと埋め込みクラスの初期化
    store_embeddings = StoreEmbeddingClass(client=client)
    retrieve_embeddings = RetrieveEmbeddingClass(client=client)
    store = Store(store_embeddings, **args.store)

    if args.create_embedding:
        store.store_embeddings("test1.pkl")
 
    store.load_embeddings("test1.pkl")
    store.old_build_faiss_index()

    # RAGの初期化
    retrieve = Retrieve(retrieve_embeddings, args.store.faiss_index_path)
    augment = AugmentPrompt(args.augmentprompt.mode)
    generate = Generate(client=client, **args.generate)

    # 質問を受け取る
    queries: list[str] = get_query()
    results = []
    
    for idx, query in enumerate(queries):
        print(f"質問{idx}: {query}")

        # 質問を入力して検索
        relevant_docs = retrieve.old_search(query=query, **args.retrieve.search)

        # プロンプトの構築
        messages = augment.build_prompt(query, relevant_docs)  

        # 回答生成
        answer = generate.generate_answer(messages)
        results.append([idx, answer])

    # csvファイルを構築する
    results_df = pd.DataFrame(results)
    results_df.to_csv('data/sample_submit/predictions.csv', index=False, header=False)


if __name__ == "__main__":
    run()