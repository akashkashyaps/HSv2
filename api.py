from typing import Union
from fastapi import FastAPI, Query
from RAGFusion_Mistral import get_rag_response

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/llm")
def process_input(input: str = Query(...)):
    result = get_rag_response(input)
    return {"input": input, "result": result}