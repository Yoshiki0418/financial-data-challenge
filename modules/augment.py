import os

class AugmentPrompt:
    """
    RAGのプロンプト構築を管理するクラス
    """

    def __init__(
        self, 
        system_content_path: str,
        user_content_path: str,
        mode: str = "default",
    ) -> None:
        """
        Args:
            mode (str, optional): 回答のモード（"default", "concise", "detailed", etc.）. Defaults to "default".
        """
        print("start AugmentPrompt")
        self.mode = mode
        self.system_content_path = system_content_path
        self.system_content = self._load_text(system_content_path)
        self.user_content_template = self._load_text(user_content_path)

    def _load_text(self, file_path: str) -> str:
        """ 指定されたテキストファイルを読み込む """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"指定されたファイルが見つかりません: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            return f.read().strip() 

    def build_prompt(self, user_query: str, doc_context: str) -> list[dict]:
        """
        プロンプトを生成する。

        Args:
            user_query (str): ユーザーの質問
            doc_context (str): 関連するコンテキスト（検索結果など）

        Returns:
            list[dict]: OpenAIのAPIに適したフォーマットのメッセージリスト
        """
        user_content = self.user_content_template.format(user_query=user_query, doc_context=doc_context)

        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": user_content}
        ]

        # messages = [
        #     {"role": "user", "content": user_content}
        # ]

        # messages = [
        #     {"role": "developer", "content": self.system_content},
        #     {"role": "user", "content": user_content}
        # ]

        return messages
