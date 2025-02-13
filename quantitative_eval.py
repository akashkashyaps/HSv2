from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import torch
import pandas as pd 
from ragas import evaluate
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    ResponseRelevancy,
    Faithfulness,
    FactualCorrectness,
    NoiseSensitivity
)
from datasets import Dataset
from ragas import EvaluationDataset
import nest_asyncio
import pandas as pd
import ast

# Apply nest_asyncio for async support
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# List of CSV files to process
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv", 
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

# Preprocess the dataset to match RAGAS expected format
def split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple rules."""
    sentences = []
    # Split by paragraphs first (newlines)
    paragraphs = text.split('\n')
    for para in paragraphs:
        # Split by sentences (period followed by space)
        if para.strip():
            for sent in para.split('. '):
                cleaned = sent.strip()
                if cleaned:
                    sentences.append(cleaned)
    return sentences

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Convert retrieved_contexts into clean sentences."""
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    # Convert string to list of Documents
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Extract sentences from each Document's page_content
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda docs: [
            sentence
            for doc in docs  # Iterate over each Document
            for sentence in split_into_sentences(doc['page_content'])
        ]
    )
    
    evaluation_dataset = EvaluationDataset.from_list(processed_df.to_dict(orient='records'))
    return evaluation_dataset

from ragas import EvaluationDataset

# List of models to evaluate
models = [
    "lly/InternLM3-8B-Instruct:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_0",
    "qwen2.5:7b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0",
    "phi3.5:3.8b-mini-instruct-q4_0",
    "mistral:7b-instruct-q4_0",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:8b-llama-distill-q4_K_M"
]

# Define the metrics to evaluate
metrics = [
    LLMContextPrecisionWithReference(),  # Context Precision
    LLMContextRecall(),                  # Context Recall
    ContextEntityRecall(),               # Context Entities Recall
    ResponseRelevancy(),                 # Response Relevancy
    Faithfulness(),                      # Faithfulness
    FactualCorrectness(),                # Factual Correctness
    NoiseSensitivity()                   # Noise Sensitivity
]

# Loop through each CSV file
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    evaluation_set = pd.read_csv(csv_file)
    dataset = preprocess_dataset(evaluation_set)

    # Loop through each model and run the evaluation
    for model_name in models:
        print(f"Starting evaluation for model: {model_name}")

        # Strong system prompt to produce *only* valid JSON
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json"
        )
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        result = evaluate(
            dataset=dataset,
            llm=llm,
            embeddings=ollama_emb,
            metrics=metrics
            )

        # Save the result if everything parsed correctly
        output_file = f"{csv_file.replace('.csv', '')}_Evaluator_{model_name}_quantitative.csv"
        result.to_pandas().to_csv(output_file, index=False)

        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")
