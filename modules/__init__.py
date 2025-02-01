# from .retrieve import Retrieve
from .store import Store
from .retrieve import Retrieve
from .augment import AugmentPrompt
from .generate import Generate
from .embedded import AzureOpenAIEmbeddings, BaseEmbeddings


__all__ = [
    "Store",
    "Retrieve",
    "AugmentPrompt",
    "Generate",
    "AzureOpenAIEmbeddings",
]

def get_embedded(embedding_type: str):
    models = {
        "AzureOpenAIEmbeddings": AzureOpenAIEmbeddings,
    }
    try:
        return models[embedding_type]
    except KeyError:
        raise ValueError(f"Unknown model name: {embedding_type}")
    
    
def create_embedding_instance(EmbeddingClass: type[BaseEmbeddings], client: any = None, model_name: str = None) -> BaseEmbeddings:
    """埋め込みクラスのインスタンスを適切な引数で生成"""
    if issubclass(EmbeddingClass, AzureOpenAIEmbeddings):
        return EmbeddingClass(client=client)
    # elif issubclass(EmbeddingClass, BERTEmbeddings):
    #     return EmbeddingClass(model_name=model_name)
    else:
        raise ValueError(f"未知の埋め込みクラス: {EmbeddingClass}")