import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN, HDBSCAN
from keybert import KeyBERT
import os
import re

# Configuration
KNOWLEDGE_BASE_FILE = "app/storage/knowledge_base.json"


def load_knowledge_base():
    """Load stored embeddings from the knowledge base JSON file."""
    with open(KNOWLEDGE_BASE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def sanitize_message(message):
    """
    Parses a message and replaces:
    - Phone numbers (10 digits) with a chain of 'x'.
    - Email addresses with 'email@adress.com'.

    Args:
        message (str): The input message to sanitize.

    Returns:
        str: The sanitized message.
    """
    # Replace phone numbers (10 consecutive digits) with '0123456789'
    message = re.sub(r'\b\d{10}\b', '0123456789', message)

    # Replace email addresses with 'email@adress.com'
    message = re.sub(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', 'email@adress.com', message)

    return message


def classify_question(question, file_path, model):
    """
    Determines whether a question is in-scope or out-of-scope based on similarity to stored knowledge.

    - Computes cosine similarity between the input and knowledge base vectors.
    - Uses a fixed threshold of 0.25 to classify the question.
    - Returns the first 200 characters of the most relevant document if in-scope.
    """

    # Sanitize the input question
    question = sanitize_message(question)

    question_vector = model.encode(question).tolist()

    knowledge_base = load_knowledge_base()
    if not knowledge_base:
        return {
            "error": "Knowledge base is empty., run embed_knowledge.py to generate embeddings."
        }

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

    if not bool(is_relevant):
        # Check if the out-of-scope file exists and initialize it if empty
        if not os.path.exists(file_path):
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Load existing out-of-scope data
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                out_of_scope_data = json.load(f)
            except json.JSONDecodeError:
                out_of_scope_data = []

        # Append the new out-of-scope question
        out_of_scope_data.append({"content": question, "vector": question_vector})

        # Save the updated out-of-scope data
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(out_of_scope_data, f, indent=4)

    return {
        "is_relevant": bool(is_relevant),
        "max_similarity": float(max_similarity),
        "threshold": float(threshold),
        "most_relevant_content": most_relevant_content,
    }


def cluster_out_of_scope_questions(
    questions_file, kw_model: KeyBERT, method="hdbscan", n_keywords=3
) -> dict:
    """
    Clusters out-of-scope questions using DBSCAN or HDBSCAN and extracts keywords using KeyBERT.

    - Uses sklearn.cluster.HDBSCAN instead of hdbscan package.
    - DBSCAN: Requires a fixed distance threshold (`eps`).
    - HDBSCAN: Adapts to variable density (recommended).
    - KeyBERT extracts top keywords from each cluster.

    Returns a dictionary mapping cluster IDs to grouped questions and their keywords.
    """
    with open(questions_file, "r", encoding="utf-8") as f:
        questions = json.load(f)

    vectors = np.array([q["vector"] for q in questions])

    if method == "hdbscan":
        cluster_model = HDBSCAN(min_cluster_size=3, metric="euclidean")
    else:
        cluster_model = DBSCAN(eps=0.3, min_samples=3, metric="cosine")

    labels = cluster_model.fit_predict(vectors)

    # Initialize data structures
    clustered_questions = {}
    cluster_keywords = {}

    # Process each question and organize clusters
    for idx, (label, question) in enumerate(zip(labels.tolist(), questions)):
        # Assign cluster ID and initialize cluster if necessary
        if label not in clustered_questions:
            clustered_questions[label] = []
        clustered_questions[label].append(question["content"])

        # Update the question with cluster ID
        question["cluster_id"] = int(label) if label != -1 else None

    # Extract keywords for each cluster
    cluster_keywords = {}
    for cluster_id, texts in clustered_questions.items():
        if cluster_id == -1:  # Skip noise points
            continue
        full_text = " ".join(texts)  # Combine texts in cluster
        keywords = kw_model.extract_keywords(
            full_text,
            keyphrase_ngram_range=(1, 2),
            stop_words="english",
            top_n=n_keywords,
        )
        cluster_keywords[cluster_id] = [kw[0] for kw in keywords]

    # Update questions with cluster keywords
    for question in questions:
        cluster_id = question["cluster_id"]
        question["cluster_keywords"] = (
            cluster_keywords.get(cluster_id, []) if cluster_id is not None else []
        )

    # Save the updated questions back to the out_of_scope_questions.json file
    with open(questions_file, "w", encoding="utf-8") as f:
        json.dump(questions, f, indent=4)

    return {
        "clusters": clustered_questions,
        "keywords": cluster_keywords,
    }
