import torch
import pandas as pd
import nest_asyncio

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from ragas import evaluate, EvaluationDataset, RunConfig
from ragas.metrics import (
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    ResponseRelevancy,
    Faithfulness,
    FactualCorrectness,
    NoiseSensitivity
)

# Apply nest_asyncio for asynchronous support
nest_asyncio.apply()

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define a callback to capture prompts and responses during evaluation.
class TestCallback(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        print("********** Prompts **********:")
        if prompts:
            print(prompts[0])
        print("\n")
    
    def on_llm_end(self, response, **kwargs):
        print("********** Response **********:")
        print(response)
        print("\n")

# List of CSV files to process.
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv", 
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

# Preprocess the dataset to match RAGAS's expected format.
def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_input": row["Question"],         # User input/query
            "retrieved_contexts": [row["Context"]],   # Retrieved contexts as a list
            "response": row["Answer"],               # Generated response
            "reference": row["Ground_Truth"]         # Reference/expected response
        })
    return EvaluationDataset.from_list(dataset)

# List of models to evaluate.
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

# Define the metrics to evaluate.
metrics = [
    LLMContextPrecisionWithReference(),  # Context Precision
    LLMContextRecall(),                  # Context Recall
    ContextEntityRecall(),               # Context Entities Recall
    ResponseRelevancy(),                 # Response Relevancy
    Faithfulness(),                      # Faithfulness
    FactualCorrectness(),                # Factual Correctness
    NoiseSensitivity()                   # Noise Sensitivity
]

# Main evaluation loop.
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    dataset = preprocess_dataset(df)

    # Loop through each model and run the evaluation.
    for model_name in models:
        print(f"\nStarting evaluation for model: {model_name}")

        # Initialise the LLM with a strict system prompt to produce only valid JSON.
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
            system=(
                "You must return a valid JSON object only. Do not include any extra text, commentary, "
                "or formatting. For example: {\"key\": \"value\"}"
            )
        )
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        # Test query to check LLM output format before full evaluation.
        test_query = "Please return a valid JSON object with a single key 'result' and a simple value."
        test_response = llm.invoke(test_query)
        print(f"Test response for model {model_name}: {test_response}")

        try:
            # Run the evaluation with the callback and a defined run configuration.
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=ollama_emb,
                raise_exceptions=True,
                callbacks=[TestCallback()],
                run_config=RunConfig(timeout=10, max_retries=1, max_wait=60, max_workers=1)
            )
        except Exception as e:
            print(f"Evaluation failed for model {model_name}: {e}")
            # Print a few dataset entries for context.
            for entry in dataset.to_pandas().to_dict(orient="records")[:5]:
                print("Debug entry:", entry)
            continue

        # Save the result if evaluation succeeds.
        output_file = f"/home/akash/HSv2/{csv_file.replace('.csv', '')}_Evaluator_{model_name}_quantitative.csv"
        result.to_pandas().to_csv(output_file, index=False)
        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")
