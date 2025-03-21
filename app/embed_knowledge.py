import json
import os
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import csv
import requests

# Configuration
DATA_DIR = "data"  # Folder containing documents to be indexed
OUTPUT_FILE = "app/storage/knowledge_base.json"
MODEL_NAME = "all-MiniLM-L6-v2"  # SBERT model for embedding generation

# Load SBERT model
model = SentenceTransformer(MODEL_NAME)


def extract_text_from_pdf(pdf_path):
    """Extracts text from a PDF using PyMuPDF (fitz)."""
    text = []
    doc = fitz.open(pdf_path)
    for page in doc:
        text.append(page.get_text("text"))
    return "\n".join(text).strip()


def embed_documents():
    """Reads knowledge_base.csv, downloads PDFs, extracts text, and generates embeddings."""
    knowledge_base = []
    csv_file = os.path.join(DATA_DIR, "knowledge_base.csv")

    if not os.path.exists(csv_file):
        print(
            f"The file {csv_file} does not exist. Please provide the knowledge_base.csv file."
        )
        return

    with open(csv_file, "r", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in tqdm(reader, desc="Processing knowledge base"):
            doc_type = row["Type de knowledge"].strip().lower()
            content = None

            if doc_type == "pdf":
                pdf_url = row["Contenu du knowledge"].strip()
                try:
                    response = requests.get(pdf_url)
                    response.raise_for_status()
                    pdf_path = os.path.join(DATA_DIR, os.path.basename(pdf_url))

                    # Save the PDF locally
                    with open(pdf_path, "wb") as pdf_file:
                        pdf_file.write(response.content)

                    # Extract text from the downloaded PDF
                    content = extract_text_from_pdf(pdf_path)
                    os.remove(pdf_path)  # Clean up the downloaded PDF after processing
                except requests.exceptions.RequestException as e:
                    print(f"⚠️ Failed to download PDF from {pdf_url}: {e}")
                    continue

            elif doc_type == "url (txt extracted column b)":
                content = row["Contenu du knowledge"].strip()

            else:
                print(f"⚠️ Unsupported document type: {doc_type}")
                continue

            if not content:
                print(
                    f"⚠️ No text extracted from {row['Contenu du knowledge']}. Skipping."
                )
                continue

            # Generate embedding
            embedding = model.encode(content).tolist()

            # Store in knowledge base
            knowledge_base.append(
                {"content": content, "type": doc_type, "vector": embedding}
            )

    # Save to JSON
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(knowledge_base, f, indent=4)

    print(f"✅ {len(knowledge_base)} documents indexed and saved in {OUTPUT_FILE}")


if __name__ == "__main__":
    embed_documents()
