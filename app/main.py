import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.utils import classify_question, cluster_out_of_scope_questions
from contextlib import asynccontextmanager
from keybert import KeyBERT
import os
# Configuration
KNOWLEDGE_BASE_FILE = "app/storage/knowledge_base.json"
OUT_OF_SCOPE_FILE = "app/storage/out_of_scope_questions.json"

ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    ml_models["sbert"] = SentenceTransformer("all-MiniLM-L6-v2")
    ml_models["kw_model"] = KeyBERT()
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

# Initialize FastAPI app
app = FastAPI(lifespan=lifespan)

class QuestionInput(BaseModel):
    question: str


@app.post("/classify")
async def classify(input_data: QuestionInput):
    """Classify whether a question is in-scope or out-of-scope."""
    vector = ml_models["sbert"].encode(input_data.question).tolist()

    result = classify_question(vector)
    print(result)
    if not result["is_relevant"]:
        # Check if the out-of-scope file exists and initialize it if empty
        if not os.path.exists(OUT_OF_SCOPE_FILE):
            with open(OUT_OF_SCOPE_FILE, "w", encoding="utf-8") as f:
                json.dump([], f)

        # Load existing out-of-scope data
        with open(OUT_OF_SCOPE_FILE, "r", encoding="utf-8") as f:
            try:
                out_of_scope_data = json.load(f)
            except json.JSONDecodeError:
                out_of_scope_data = []

        # Append the new out-of-scope question
        out_of_scope_data.append({"content": input_data.question, "vector": vector})

        # Save the updated out-of-scope data
        with open(OUT_OF_SCOPE_FILE, "w", encoding="utf-8") as f:
            json.dump(out_of_scope_data, f, indent=4)

    return {"classification": result}


@app.get("/clusters")
async def get_clusters():
    """Return clusters of out-of-scope questions."""
    with open(OUT_OF_SCOPE_FILE, "r", encoding="utf-8") as f:
        out_of_scope_data = json.load(f)

    clusters = cluster_out_of_scope_questions(out_of_scope_data, ml_models["kw_model"])

    return {"clusters": clusters}
