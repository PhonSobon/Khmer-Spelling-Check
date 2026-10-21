"""HTTP API for the Khmer spell checker (after training).

    python main.py --serve        (or: uvicorn app:app --port 8000)
    open http://localhost:8000/docs

POST /spell    {"text": "ខ្ញុំជាសស្សដែលមានសក្សានៅកម្ពុច", "top_k": 3}
               -> each misspelled word, its position and its top-k suggestions
POST /suggest  {"word": "សំណើរ", "left": "", "right": "", "top_k": 3}
               -> top-k suggestions for one word (context optional)
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from khspell.topk import Speller

app = FastAPI(title="Khmer Spell Checker")
try:
    speller = Speller.load()
except Exception as e:   # not trained yet
    speller = None
    print(f"spell checker not loaded ({e}). Train it first: python main.py")


class TextRequest(BaseModel):
    text: str
    top_k: int = 3
    threshold: float = 0.5   


class WordRequest(BaseModel):
    word: str
    left: str = ""
    right: str = ""
    top_k: int = 3


def _ready():
    if speller is None:
        raise HTTPException(503, "model not trained yet: run  python main.py")


@app.post("/spell")
def spell(req: TextRequest):
    _ready()
    text, errors = speller.check(req.text, k=req.top_k, tag_threshold=req.threshold)
    return {"text": text, "errors": errors}


@app.post("/suggest")
def suggest(req: WordRequest):
    _ready()
    return {"word": req.word, "suggestions": speller.suggest(req.word, req.left, req.right, k=req.top_k)}
