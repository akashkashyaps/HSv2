from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
import torch
import pandas as pd 
from ragas import evaluate
from ragas.metrics import (
    BleuScore,
    RougeScore,
    StringPresence
)
from ragas.metrics._string import NonLLMStringSimilarity
from datasets import Dataset
import nest_asyncio

# Apply nest_asyncio for async support
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# List of CSV files to process
csv_files = [
    "evaluation_set_FusionPhi3.csv",
    "evaluation_set_FusionMistral.csv",
    "evaluation_set_FusionLlama3.csv"
]

# Preprocess the dataset to match RAGAS expected format
def preprocess_dataset(df):
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "question": row["question"],  # User input/query
            "contexts": [row["context"]],  # Retrieved contexts (as a list)
            "answer": row["answer"],  # Generated response
            "ground_truth": row["ground_truth"]  # Reference/expected response
        })
    return Dataset.from_pandas(pd.DataFrame(dataset))


# Define the metrics to evaluate
metrics = [
    BleuScore,  
    RougeScore,  
    StringPresence,  
    NonLLMStringSimilarity
]

# Loop through each CSV file
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")

    # Load the evaluation dataset
    evaluation_set = pd.read_csv(csv_file)

    # Convert the dataset to the required format
    dataset = preprocess_dataset(evaluation_set)

    # Loop through each model and run the evaluation
    for model_name in models:
        print(f"Starting evaluation for model: {model_name}")

        # Load the model and embeddings for each run
        llm = ChatOllama(
            model=model_name,
            temperature=0.6)  # Initialize the model for each iteration
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        # Run RAGAS evaluation
        result = evaluate(
            dataset=dataset,
            llm=llm,
            embeddings=ollama_emb,
            metrics=metrics
        )

        # Save the result to a CSV file with the model name and dataset name in the file name
        output_file = f"{csv_file.replace('.csv', '')}_Evaluator_{model_name}_nonLLM.csv"
        result.to_pandas().to_csv(output_file, index=False)
        
        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")