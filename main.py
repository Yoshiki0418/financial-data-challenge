import hydra
from omegaconf import DictConfig

from modules.get_query import get_query

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):

    # 質問を受け取るための関数
    queries: list[str] = get_query()
    
    for idx, query in enumerate(queries):
        print(f"{idx}: {query}")

        # 質問を入力して検索

        # プロンプトの構築

        # 回答生成

        # csvファイルを構築する

    pass


if __name__ == "__main__":
    run()