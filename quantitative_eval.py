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
from ragas import EvaluationDataset
import nest_asyncio

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
def preprocess_dataset(df):
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_input": row["Question"],        # User input/query
            "retrieved_contexts": [row["Context"]],  # Retrieved contexts as a list
            "response": row["Answer"],            # Generated response
            "reference": row["Ground_Truth"]      # Reference/expected response
        })
    
    return EvaluationDataset.from_list(dataset)


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
            format="json",
            system="You must strictly return a valid JSON format. Do not include any explanations or additional text."
        )
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        try:
            result = evaluate(
                dataset=dataset,
                llm=llm,
                embeddings=ollama_emb,
                metrics=metrics
            )
        except Exception as e:
            print(f"Evaluation failed for model {model_name}: {e}")
    
        # Debug: Capture raw LLM outputs
            for entry in dataset.to_pandas().to_dict(orient="records")[:5]:  # Show first 5 entries
                print("Debugging entry:", entry)
    
            continue


        # Save the result if everything parsed correctly
        output_file = f"/home/akash/HSv2/{csv_file.replace('.csv', '')}_Evaluator_{model_name}_quantitative.csv"
        result.to_pandas().to_csv(output_file, index=False)

        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")
