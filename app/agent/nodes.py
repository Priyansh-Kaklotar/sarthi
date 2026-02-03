from app.agent.state import SarthiState
from app.agent.tools import retrieve_gita_docs
from langchain_core.prompts import ChatPromptTemplate
from app.config import SYSTEM_PROMPT_PATH, ANSWER_PROMPT_PATH
import ollama


llm = ollama.Client(host="http://localhost:11434")
MODEL_NAME = "llama3"


with open(str(SYSTEM_PROMPT_PATH), "r", encoding="utf-8") as f:
    system_prompt = f.read()

with open(str(ANSWER_PROMPT_PATH), "r", encoding="utf-8") as f:
    answer_prompt = f.read()


# -------------------------
# Retrieve Node
# -------------------------
def retrive_node(state: SarthiState) -> SarthiState:
    question = state["question"]
    docs = retrieve_gita_docs(question)

    return {
        "question": question,
        "documents": docs,
        "answer": None
    }


# -------------------------
# Answer Node
# -------------------------
def answer_node(state: SarthiState) -> SarthiState:
    docs = state["documents"]

    if not docs:
        return {
            **state,
            "answer": "O seeker, this question is not answered directly in the sacred verses provided."
        }

    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", answer_prompt),
    ])

    formatted_prompt = prompt.format(
        context=context,
        question=state["question"]
    )

    response = llm.generate(
        model=MODEL_NAME,
        prompt=formatted_prompt
    )

    return {
        **state,
        "answer": response["response"]
    }
