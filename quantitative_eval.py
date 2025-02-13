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
import re

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
def extract_page_contents(text: str) -> list:
    """
    Extracts the content of each page from a string containing one or more 
    Document(...) objects. This function looks for the pattern:
      page_content="...text..."
    and returns a list where each element is the text for one page.
    """
    # This pattern captures everything between page_content=" and the next "
    pattern = r'page_content="(.*?)"'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    # Return each captured page content as a single, stripped string
    return [match.strip() for match in matches if match.strip()]

def preprocess_dataset(df: pd.DataFrame):
    """
    Prepares dataset for evaluation by renaming columns and extracting
    the page_content from the retrieved contexts into a list of strings,
    where each element corresponds to one page's content.
    """
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    # For each row in 'retrieved_contexts', if it is a string, extract the page contents.
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: extract_page_contents(x) if isinstance(x, str) else x
    )
    
    # Each row now has retrieved_contexts as a list of strings,
    # where the first element is the first page context, the second is the second, etc.
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
from ragas.run_config import RunConfig
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
            format="json",
            num_ctx= 10000,
            system= "You are a helpful assistant that follows directions acording to the provided schema. Your response must be a valid JSON object with no additional commentary or text. Do not output any explanation or extra text; only output a valid JSON."
        )
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        result = evaluate(
            dataset=dataset,
            llm=llm,
            embeddings=ollama_emb,
            metrics=metrics,
            run_config=RunConfig(max_retries=5,timeout=600)
            )

        # Save the result if everything parsed correctly
        output_file = f"{csv_file.replace('.csv', '')}_Evaluator_{model_name}_quantitative.csv"
        result.to_pandas().to_csv(output_file, index=False)

        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")
