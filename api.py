from typing import Union
from fastapi import FastAPI, Query
from RAG_Groq import groq_response

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/llm")
def process_input(input: str = Query(...)):
    result = groq_response(input)
    return {"input": input, "result": result}