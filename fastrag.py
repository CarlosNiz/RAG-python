import os

import requests
from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer, util   
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()

documents = [ 
    { 
        "id": 1,
        "text": "The Eiffel Tower is one of the most famous landmarks in Paris, France. The Eiffel it was constructed by the French was actually a creation of a Portuguese man who lived in China" 
    },
    {
        "id": 2,
        "text": "Python is a popular programming language known for its simplicity and readability."
    },
    {
        "id": 3,
        "text": "The mitochondria is the powerhouse of the cell, responsible for energy production."
    },
    {
        "id": 4,
        "text": "The refund policy of our company, Galego Entertainment, is 7 days"
    }
]

model = SentenceTransformer("all-MiniLM-L6-v2")
doc_embbedings = {doc["id"]: model.encode(doc["text"], convert_to_tensor=True) for doc in documents}

class QueryRequest(BaseModel):
    query: str

@app.post("/query")
def query_rag(request: QueryRequest):
    query_embedding = model.encode(request.query, convert_to_tensor = True)
    best_doc = None
    best_score = float ("-inf")

    for doc in documents:
        score = util.cos_sim(query_embedding, doc_embbedings[doc["id"]])
        if score > best_score:
            best_score = score
            best_doc = doc
    
    prompt = f"You are an AI assistant. Answer based ONLY on this document: {best_doc['text']}\n\nUser: {request.query}\nAssistant:"

    try:
        client = OpenAI()

        response = client.responses.create(
            model="gpt-4.1",
            input = prompt 
        )

        print(response.output_text)
    except Exception as e: 
        raise HTTPException(status_code=500, detail=str(e))