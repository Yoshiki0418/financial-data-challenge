import pandas as pd


def get_query() -> list[str]:
    """
    質問をcsvファイルから受け取り、リスト形式で変換する
    """
    df = pd.read_csv("data/query.csv") 
    
    queries: list[str] = df["problem"].tolist()
    return queries
