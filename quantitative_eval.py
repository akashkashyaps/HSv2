from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings
import torch
import pandas as pd 

import nest_asyncio
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# llm = Ollama(model="mistral")  

# ollama_emb = OllamaEmbeddings(
#     model="nomic-embed-text",
# )  


from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness,
    answer_similarity
)
# from ragas.metrics.critique import harmfulness
# from ragas.metrics.critique import maliciousness
# from ragas.metrics.critique import coherence
# from ragas.metrics.critique import conciseness
from ragas import evaluate
from datasets import Dataset

# def evaluate_ragas_dataset(ragas_dataset):
#   result = evaluate(
#     ragas_dataset,
#     llm=llm,
#     embeddings=ollama_emb,
#     raise_exceptions=False,
#     metrics=[
#         faithfulness,
#         answer_relevancy,
#         answer_correctness,
#         answer_similarity
#     ],
#   )
#   return result


# # def qualitative_analysis(ragas_dataset):
# #   result = evaluate(
# #     ragas_dataset,
# #     llm=llm,
# #     embeddings=ollama_emb,
# #     raise_exceptions=False,
# #     metrics=[
# #         harmfulness,
# #         maliciousness,
# #         coherence,
# #         conciseness
# #     ],
# #   )
# #   return result



# evaluation_set = pd.read_csv("evaluation_set_FusionLlama3.csv")
# # evaluation_set = evaluation_set.head(5)
# # Convert the context column to a list of strings
# evaluation_set['context'] = evaluation_set['context'].apply(lambda x: [x])
# evaluation_set.drop(columns=["contexts"], inplace=True)
# evaluation_set.rename(columns={"context": "contexts"}, inplace=True)


# from datasets import Dataset
# dataset = Dataset.from_pandas(evaluation_set)

# quantitative_result_qwen = evaluate_ragas_dataset(dataset)
# # qualitative_result_qwen = qualitative_analysis(dataset)
# quantitative_result_qwen.to_pandas().to_csv("Base_FusionLlama3-Evaluator_mistral-quantitative.csv", index=False)
# # qualitative_result_qwen.to_pandas().to_csv("Base_Mistral7B-Evaluator_Qwen-qualitative.csv", index=False)

# Function to evaluate the dataset using a given model
def evaluate_ragas_dataset(ragas_dataset, llm, embeddings):
    result = evaluate(
        ragas_dataset,
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
        metrics=[
            faithfulness,
            answer_relevancy,
            answer_correctness,
            answer_similarity
        ],
    )
    return result

# Load the evaluation dataset
evaluation_set = pd.read_csv("evaluation_set_FusionLlama3.csv")
evaluation_set['context'] = evaluation_set['context'].apply(lambda x: [x])
evaluation_set.drop(columns=["contexts"], inplace=True)
evaluation_set.rename(columns={"context": "contexts"}, inplace=True)

# Convert to Hugging Face Dataset format
dataset = Dataset.from_pandas(evaluation_set)

# List of models to evaluate
models = ["internlm2", "llama3", "qwen2", "gemma2", "phi3"]  

# Loop through each model and run the evaluation
for model_name in models:
    print(f"Starting evaluation for model: {model_name}")

    # Load the model and embeddings for each run
    llm = Ollama(model=model_name)  # Initialize the model for each iteration
    ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

    # Run RAGAS evaluation
    quantitative_result = evaluate_ragas_dataset(dataset, llm, ollama_emb)
    
    # Save the result to a CSV file with the model name in the file name
    output_file = f"Base_FusionLlama3-Evaluator_{model_name}-quantitative.csv"
    quantitative_result.to_pandas().to_csv(output_file, index=False)
    
    print(f"Completed evaluation for model: {model_name}")
    print(f"Results saved to: {output_file}")
