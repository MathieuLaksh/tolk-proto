import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, HDBSCAN
from keybert import KeyBERT

# Configuration
KNOWLEDGE_BASE_FILE = "app/storage/knowledge_base.json"
THRESHOLD_PERCENTILE = 90  # Adaptive thresholding


def load_knowledge_base():
    """Load stored embeddings from the knowledge base JSON file."""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def classify_question(question_vector):
    """
    Determines whether a question is in-scope or out-of-scope based on similarity to stored knowledge.
    
    - Computes cosine similarity between the input and knowledge base vectors.
    - Uses a fixed threshold of 0.8 to classify the question.
    - Returns the first 200 characters of the most relevant document if in-scope.
    """
    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return {"error": "Knowledge base is empty."}

    vectors = np.array([doc["vector"] for doc in knowledge_base])

    # Compute cosine similarities
    similarities = cosine_similarity([question_vector], vectors)[0]
    print(similarities)

    # Fixed threshold
    threshold = 0.25

    # Get the highest similarity score and its index
    max_similarity = np.max(similarities)
    max_index = np.argmax(similarities)

    is_relevant = max_similarity >= threshold

    # Retrieve the first 200 characters of the most relevant document if in-scope
    most_relevant_content = (
        knowledge_base[max_index]["content"][:200] if is_relevant else None
    )

    return {
        "is_relevant": bool(is_relevant),
        "max_similarity": float(max_similarity),
        "threshold": float(threshold),
        "most_relevant_content": most_relevant_content,
    }


def cluster_out_of_scope_questions(questions, kw_model: KeyBERT, method="hdbscan", n_keywords=3) -> dict:
    """
    Clusters out-of-scope questions using DBSCAN or HDBSCAN and extracts keywords using KeyBERT.

    - Uses sklearn.cluster.HDBSCAN instead of hdbscan package.
    - DBSCAN: Requires a fixed distance threshold (`eps`).
    - HDBSCAN: Adapts to variable density (recommended).
    - KeyBERT extracts top keywords from each cluster.

    Returns a dictionary mapping cluster IDs to grouped questions and their keywords.
    """
    vectors = np.array([q["vector"] for q in questions])

    if method == "hdbscan":
        cluster_model = HDBSCAN(min_cluster_size=3, metric="euclidean")
    else:
        cluster_model = DBSCAN(eps=0.3, min_samples=3, metric="cosine")

    labels = cluster_model.fit_predict(vectors)

    # Organize questions into clusters
    clustered_questions = {}
    for idx, label in enumerate(labels.tolist()):
        if label not in clustered_questions:
            clustered_questions[label] = []
        clustered_questions[label].append(questions[idx]["content"])

    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id, texts in clustered_questions.items():
        full_text = " ".join(texts)  # Combine texts in cluster
        keywords = kw_model.extract_keywords(full_text, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=n_keywords)
        cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

    return {
        "clusters": clustered_questions,
        "keywords": cluster_keywords,
    }

