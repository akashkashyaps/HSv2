from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings
import torch
import pandas as pd 

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

llm = Ollama(model="llama3")  

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)  


from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)
from ragas.metrics.critique import harmfulness
from ragas import evaluate
from datasets import Dataset

def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    llm=llm,
    embeddings=ollama_emb,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

# import ast  # Import the ast module to safely evaluate string representations of lists

# def preprocess_data(data):
#     preprocessed_data = []
#     for _, row in data.iterrows():
#         contexts = ast.literal_eval(row["contexts"])  # Convert the string representation to a list
#         preprocessed_data.append({
#             "question": row["question"],
#             "context": row["context"],
#             "ground_truth": row["answer"],
#             "contexts": contexts  # Use the converted list as the "contexts" field
#         })
#     return preprocessed_data

evaluation_set = pd.read_csv("evaluation_set.csv")
# Convert the context column to a list of strings
evaluation_set['context'] = evaluation_set['context'].apply(lambda x: [x])
evaluation_set

# preprocessed_data = preprocess_data(evaluation_set)

# from datasets import Dataset

# data = [
#     {
#         "question": row["question"],
#         "context": row["context"],
#         "ground_truth": row["answer"],
#         "contexts": row["contexts"].split("\n") 
#     }
#     for _, row in evaluation_set.iterrows()
# ]

# dataset = Dataset.from_dict(data)


qa_result = evaluate_ragas_dataset(evaluation_set)
qa_result.to_csv("qa_result.csv", index=False)




# from datasets import load_dataset
# eng = load_dataset("explodinggradients/amnesty_qa","english")
# eng["eval"].to_csv("eng.csv", index=False)


# import pandas as pd
# eng_data = pd.read_csv("eng.csv")

# eng_data.head(1)["contexts"].values[0]