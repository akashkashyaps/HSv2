from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings
import torch
import pandas as pd 

import nest_asyncio
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

llm = Ollama(model="mistral")  

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)  


from ragas.metrics.critique import harmfulness
from ragas.metrics.critique import maliciousness
from ragas.metrics.critique import coherence
from ragas.metrics.critique import conciseness
from ragas import evaluate
from datasets import Dataset


def qualitative_analysis(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    llm=llm,
    embeddings=ollama_emb,
    raise_exceptions=False,
    metrics=[
        harmfulness,
        maliciousness,
        coherence,
        conciseness
    ],
  )
  return result



evaluation_set = pd.read_csv("evaluation_set_FusionLlama3.csv")
# Convert the context column to a list of strings
evaluation_set['context'] = evaluation_set['context'].apply(lambda x: [x])
evaluation_set.drop(columns=["contexts"], inplace=True)
evaluation_set.rename(columns={"context": "contexts"}, inplace=True)


from datasets import Dataset
dataset = Dataset.from_pandas(evaluation_set)


qualitative_result_qwen = qualitative_analysis(dataset)
qualitative_result_qwen.to_pandas().to_csv("Base_FusionLlama3-Evaluator_mistral-qualitative.csv", index=False)
