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
def clean_context_text(text: str) -> list:
    """
    Splits 'text' by newlines, strips whitespace,
    and filters out empty lines — one simple approach to chunking.
    """
    lines = text.split('\n')
    lines = [line.strip() for line in lines if line.strip()]
    return lines

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepares dataset for evaluation by renaming columns and
    simplifying 'retrieved_contexts' into lists of strings.
    """
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    # Convert string representation of list ("[Doc1, Doc2]") to an actual Python list
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    
    # Optionally split each doc in the retrieved_contexts
    # so we have smaller pieces of text (list of lines).
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda doc_list: [clean_context_text(doc) for doc in doc_list]
    )
    
    # Flatten each row so we end up with a single list of lines
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda doc_list: [line for sublist in doc_list for line in sublist]
    )
    
    evaluation_dataset = EvaluationDataset.from_pandas(processed_df)
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
