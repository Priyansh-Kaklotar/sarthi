from app.rag.embeddings import get_embedding_model
from langchain_chroma import Chroma
from app.config import CHROMA_PATH


def load_vectorstore():
    embeddings = get_embedding_model()

    return Chroma(
        persist_directory=str(CHROMA_PATH),
        embedding_function=embeddings
    )


def get_retriever(
    k: int = 6,
    search_type: str = "similarity",
    filters: dict | None = None,
    fetch_k: int | None = None,
    lambda_mult: float | None = None
):
    vectordb = load_vectorstore()
    search_kwargs = {"k": k}

    if filters:
        # 🔥 AUTO-CONVERT simple dict → Chroma $and syntax
        if len(filters) > 1:
            search_kwargs["filter"] = {
                "$and": [{k: v} for k, v in filters.items()]
            }
        else:
            key, value = next(iter(filters.items()))
            search_kwargs["filter"] = {key: value}

    if fetch_k is not None:
        search_kwargs["fetch_k"] = fetch_k

    if lambda_mult is not None:
        search_kwargs["lambda_mult"] = lambda_mult

    return vectordb.as_retriever(
        search_type=search_type,
        search_kwargs=search_kwargs
    )
