import sys
sys.path.append("/usr/local/lib/python3.10/dist-packages")
import tiktoken
from typing import List, Dict
from openai import OpenAI

class Generate:
    def __init__(
        self, 
        client: OpenAI, 
        model: str = "gpt-4o", 
        max_tokens: int = 54, 
        temperature: float = 0.7
    ) -> None:
        """
        回答生成クラス

        Args:
            client (Any): Azure OpenAI API のクライアント
            model (str): 使用するモデル（デフォルト: "gpt-4o"）
            max_tokens (int): 応答に使える最大トークン数
            temperature (float): 創造性の制御（0~1）
        """
        self.client = client
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.encoding = tiktoken.encoding_for_model("gpt-4o")

    def _count_tokens(self, text: str) -> int:
        """テキストのトークン数を計算する"""
        tokens = self.encoding.encode(text)
        return len(tokens)

    def _truncate_to_max_tokens(self, text: str) -> str:
        """トークン数が上限を超えた場合に切り詰める"""
        tokens = self.encoding.encode(text)
        truncated_tokens = tokens[:self.max_tokens]
        return self.encoding.decode(truncated_tokens)

    def generate_answer(self, messages: List[Dict[str, str]]) -> str:
        """
        AI による回答生成

        Args:
            messages (List[Dict[str, str]]): AugmentPrompt で作成されたメッセージリスト

        Returns:
            str: 生成された回答
        """
        roop_count = 0
        while True:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=1,
                frequency_penalty=0,
                presence_penalty=0,
                stream=False
            )

            # response = self.client.chat.completions.create(
            #     model="o1",
            #     messages=messages,
            #     max_tokens=self.max_tokens, 
            #     temperature=self.temperature,  
            #     stream=False
            # )
            
            answer = response.choices[0].message.content.strip() if response.choices and response.choices[0].message.content else "わかりません。"

            answer = answer.replace("\n", " ").replace("\r", " ")
            
            token_count = self._count_tokens(answer)

            if token_count <= self.max_tokens:
                break  # トークン数が条件内に収まったら終了

            if roop_count == 2:
                # 強制的に max_tokens に収まるようカットする
                answer = self._truncate_to_max_tokens(answer)
                break

            print(f"トークン数 {token_count} が {self.max_tokens} を超えたため再生成します")
            roop_count += 1

        print(f"答え:{answer}")
        return answer
