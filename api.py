from typing import Union
from fastapi import FastAPI, Query
from RAG_TTS_Ollama_Edge import get_rag_response_ollama

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/llm")
def process_input(input: str = Query(...)):
    result = get_rag_response_ollama(input)
    return {"input": input, "result": result}