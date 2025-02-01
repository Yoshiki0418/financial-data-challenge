# from .retrieve import Retrieve
from .store import Store
from .retrieve import Retrieve
from .augment import AugmentPrompt
from .generate import Generate
from .embedded import AzureOpenAIEmbeddings


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