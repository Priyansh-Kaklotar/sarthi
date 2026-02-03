from typing import List, Optional, TypedDict
from langchain_core.documents import Document


class SarthiState(TypedDict):
    question: str
    documents: List[Document]
    answer: Optional[str]
