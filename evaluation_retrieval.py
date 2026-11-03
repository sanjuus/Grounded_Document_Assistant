
from evaluation_dataset import evaluation_questions
from rag_pipeline import (
    load_embedding_model,
    process_documents,
    hybrid_retrieve
)

# Load the PDF used for evaluation
PDF_PATH = "Astrophysical Horizons Reference Compendium.pdf"

# Load embedding model
model = load_embedding_model()

# Process the PDF and build retrieval indexes
# Process the PDF and build retrieval indexes
with open(PDF_PATH, "rb") as pdf_file:
    chunks, faiss_index, bm25_index = process_documents([pdf_file], model)

# Evaluation settings
K_VALUES = [5, 10]

results = []

for item in evaluation_questions:

    question = item["question"]

    # Retrieve the top 10 chunks
    retrieved_chunks = hybrid_retrieve(
        question,
        model,
        faiss_index,
        bm25_index,
        chunks,
        top_k=10
    )

    # Check whether expected source and page were retrieved
    relevant_found = {}

    for k in K_VALUES:
        relevant_found[k] = any(
            chunk["source"] == item["expected_source"]
            and chunk["page"] in item["expected_pages"]
            for chunk in retrieved_chunks[:k]
        )

    results.append({
        "id": item["id"],
        "question": question,
        "recall_at_5": relevant_found[5],
        "recall_at_10": relevant_found[10]
    })

# Calculate overall Recall@K
total_questions = len(results)

recall_5 = sum(
    result["recall_at_5"] for result in results
) / total_questions

recall_10 = sum(
    result["recall_at_10"] for result in results
) / total_questions

# Display results
print("\n========== RETRIEVAL EVALUATION ==========")

for result in results:
    print(f"\n{result['id']}: {result['question']}")
    print(f"Recall@5 hit: {result['recall_at_5']}")
    print(f"Recall@10 hit: {result['recall_at_10']}")

print("\n========== FINAL METRICS ==========")
print(f"Recall@5:  {recall_5:.2%}")
print(f"Recall@10: {recall_10:.2%}")
