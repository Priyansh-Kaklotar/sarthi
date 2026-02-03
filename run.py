from app.agent.state import SarthiState
from app.agent.nodes import retrive_node , answer_node

state = SarthiState(
    question="What is life?",
    documents=[],
    answer=None
)

state = retrive_node(state)
state = answer_node(state)

print(state["answer"])