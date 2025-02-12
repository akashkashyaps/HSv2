import torch
import pandas as pd
import nest_asyncio

from mlx_lm import load, generate
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

##################################################
# 1) Preprocessing the dataset for RAGAS
##################################################
def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_input": row["Question"],            # User input/query
            "retrieved_contexts": [row["Context"]],   # Retrieved contexts as a list
            "response": row["Answer"],                # Generated response
            "reference": row["Ground_Truth"]          # Reference/expected response
        })
    return EvaluationDataset.from_list(dataset)

##################################################
# 2) Define the metrics
##################################################
metrics = [
    LLMContextPrecisionWithReference(),  # Context Precision
    LLMContextRecall(),                  # Context Recall
    ContextEntityRecall(),               # Context Entities Recall
    ResponseRelevancy(),                 # Response Relevancy
    Faithfulness(),                      # Faithfulness
    FactualCorrectness(),                # Factual Correctness
    NoiseSensitivity()                   # Noise Sensitivity
]

##################################################
# 3) List of CSV files to evaluate
##################################################
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv",
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

##################################################
# 4) List of MLX models
##################################################
models = [
    "mlx-community/internlm3-8b-instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "mlx-community/Phi-3.5-mini-instruct-4bit",
    "mlx-community/gemma-2-9b-it-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-1M-4bit",
    "mlx-community/Meta-Llama-3.1-8B-Instruct-4bit"
]

##################################################
# 5) Main evaluation loop (no callbacks used)
##################################################
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        continue

    # Convert CSV to RAGAS dataset
    dataset = preprocess_dataset(df)

    for model_name in models:
        print(f"\nStarting evaluation for MLX model: {model_name}")

        # Load model & tokenizer from mlx-community
        model, tokenizer = load(model_name)

        # Prepare an LLM-like function for RAGAS to call
        def llm_fn(input_text: str) -> str:
            # If the tokenizer has a chat template, apply it
            if tokenizer.chat_template is not None:
                messages = [{"role": "user", "content": input_text}]
                prompt = tokenizer.apply_chat_template(
                    messages, add_generation_prompt=True
                )
            else:
                prompt = input_text
            # Generate the output
            return generate(model, tokenizer, prompt=prompt, verbose=False).strip()

        try:
            # Evaluate with RAGAS
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm_fn,                 # Use our custom function for generation
                embeddings=None,            # If you need embeddings, provide them here
                raise_exceptions=True,
                run_config=RunConfig(
                    timeout=10, 
                    max_retries=1, 
                    max_wait=60, 
                    max_workers=1
                )
            )

            # Save the evaluation results
            output_file = (
                f"/home/akash/HSv2/"
                f"{csv_file.replace('.csv', '')}_Evaluator_"
                f"{model_name.replace('/', '_')}_quantitative.csv"
            )
            result.to_pandas().to_csv(output_file, index=False)
            print(f"Completed evaluation for model: {model_name}")
            print(f"Results saved to: {output_file}")

        except Exception as e:
            print(f"Evaluation failed for model {model_name}: {e}")
            # Optionally print a few dataset entries to debug
            for entry in dataset.to_pandas().head(3).to_dict(orient="records"):
                print("Debug entry:", entry)

    print(f"Finished processing dataset: {csv_file}")
