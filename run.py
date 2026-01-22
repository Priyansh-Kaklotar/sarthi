from app.rag.loader import load_gita_documents

docs = load_gita_documents()
print(len(docs))
print(docs[0])