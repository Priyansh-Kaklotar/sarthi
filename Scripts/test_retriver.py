from app.rag.retriever import get_retriever

def main():
    retriever = get_retriever(
        k=5,
        filters={
            "type": "commentary",
            "language": "english"
        }
    )

    query = "what is karma yoga?"
    docs = retriever.invoke(query)

    print("\nQuery:", query)
    print("Retrieved docs:", len(docs))
    print("=" * 80)

    for i, doc in enumerate(docs, start=1):
        print(f"\n--- Document {i} ---")
        print("Metadata:", doc.metadata)
        print("Content preview:")
        print(doc.page_content[:400])
        print("-" * 80)

if __name__ == "__main__":
    main()
