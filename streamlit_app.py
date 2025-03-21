import json
import numpy as np
import streamlit as st
import umap
import matplotlib.pyplot as plt

# Configuration
OUT_OF_SCOPE_FILE = "app/storage/out_of_scope_questions.json"

# Load out-of-scope questions
def load_out_of_scope_questions(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Streamlit app
def main():
    st.title("Out-of-Scope Questions Visualization")

    # Load data
    st.sidebar.header("Configuration")
    file_path = st.sidebar.text_input("Out-of-Scope File Path", OUT_OF_SCOPE_FILE)
    if not file_path:
        st.error("Please provide the path to the out-of-scope questions file.")
        return

    try:
        questions = load_out_of_scope_questions(file_path)
    except FileNotFoundError:
        st.error(f"File not found: {file_path}")
        return
    except json.JSONDecodeError:
        st.error(f"Invalid JSON format in file: {file_path}")
        return

    # Extract vectors and metadata
    vectors = np.array([q["vector"] for q in questions])
    contents = [q["content"] for q in questions]
    cluster_ids = [q.get("cluster_id", None) for q in questions]

    # UMAP dimensionality reduction
    st.sidebar.subheader("UMAP Parameters")
    n_neighbors = st.sidebar.slider("Number of Neighbors", 2, 50, 15)
    min_dist = st.sidebar.slider("Minimum Distance", 0.0, 1.0, 0.1)
    n_components = st.sidebar.selectbox("Number of Dimensions", [2, 3], index=0)

    st.write("Performing UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=n_components, random_state=42)
    embedding = reducer.fit_transform(vectors)
    
    unique_ids = np.unique(np.array(cluster_ids)[np.array(cluster_ids) != np.array(None)])
    print(unique_ids)
    # Plot the results
    st.write("### UMAP Visualization")
    if n_components == 2:
        fig, ax = plt.subplots(figsize=(10, 8))
        cmap = plt.cm.get_cmap("tab10", len(unique_ids) + 1)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=cluster_ids, cmap=cmap, s=50, alpha=0.7)
        ax.set_title("2D UMAP Projection of Out-of-Scope Questions")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        st.pyplot(fig)
    elif n_components == 3:
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        cmap = plt.cm.get_cmap("tab10", len(unique_ids) + 1)
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c=cluster_ids, cmap=cmap, s=50, alpha=0.7)
        ax.set_title("3D UMAP Projection of Out-of-Scope Questions")
        ax.set_xlabel("UMAP Dimension 1")
        ax.set_ylabel("UMAP Dimension 2")
        ax.set_zlabel("UMAP Dimension 3")
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        st.pyplot(fig)

    # Display question details
    st.write("### Question Details")
    for i, content in enumerate(contents):
        st.write(f"**Question {i + 1}:** {content}")
        st.write(f"Cluster ID: {cluster_ids[i]}")
        st.write("---")

if __name__ == "__main__":
    main()