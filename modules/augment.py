class AugmentPrompt:
    """
    RAGのプロンプト構築を管理するクラス
    """

    def __init__(self, mode: str = "default"):
        """
        Args:
            mode (str, optional): 回答のモード（"default", "concise", "detailed", etc.）. Defaults to "default".
        """
        print("start AugmentPrompt")
        self.mode = mode
        self.system_content = self._get_system_content()

    def _get_system_content(self) -> str:
        """システムメッセージを取得（モードに応じて変更可能）"""
        base_content = (
            "あなたは優秀なAIアシスタントです。\n"
            "ユーザーが与えた情報だけをもとに回答してください。\n"
            "情報がコンテキストに含まれない場合は『わかりません』と答えてください。\n"
            "数量で答える問題の回答には、質問に記載の単位を使うこと。\n"
            "問題に過不足なく回答すること。\n"
        )

        if self.mode == "concise":
            base_content += "回答は簡潔にしてください。\n"
        elif self.mode == "detailed":
            base_content += "回答はできるだけ詳細に説明してください。\n"
        elif self.mode == "formal":
            base_content += "フォーマルな言葉遣いで回答してください。\n"
        
        return base_content

    def build_prompt(self, user_query: str, doc_context: str) -> list[dict]:
        """
        プロンプトを生成する。

        Args:
            user_query (str): ユーザーの質問
            doc_context (str): 関連するコンテキスト（検索結果など）

        Returns:
            list[dict]: OpenAIのAPIに適したフォーマットのメッセージリスト
        """
        user_content = (
            "以下のコンテキストを参考に回答してください。\n"
            f"質問:\n{user_query}\n\n"
            "コンテキスト:\n"
            f"{doc_context}"
        )

        messages = [
            {"role": "system", "content": self.system_content},
            {"role": "user", "content": user_content}
        ]

        return messages
