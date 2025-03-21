from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from app.utils import classify_question, cluster_out_of_scope_questions
from contextlib import asynccontextmanager
from keybert import KeyBERT

# Configuration
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

    result = classify_question(
        input_data.question, OUT_OF_SCOPE_FILE, ml_models["sbert"]
    )
    print(result)

    return {"classification": result}


@app.get("/clusters")
async def get_clusters():
    """Return clusters of out-of-scope questions."""

    clusters = cluster_out_of_scope_questions(OUT_OF_SCOPE_FILE, ml_models["kw_model"])

    return {"clusters": clusters}
