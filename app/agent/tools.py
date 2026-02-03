from typing import List
from langchain_core.documents import Document
from app.rag.retriever import get_retriever

def retrieve_gita_docs(question :str)->List[Document]:
    """
    Docstring for retrieve_gita_docs
    
    :param question: Description
    :type question: str
    :param vectorstore: vectorestore that helps to retreive document
    :return: Description
    :rtype: List[Document]

    """
    retrieve = get_retriever(k=3)
    return retrieve.invoke(question)