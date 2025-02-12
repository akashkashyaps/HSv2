import torch
import pandas as pd
import nest_asyncio

from transformers import AutoTokenizer, AutoModelForCausalLM
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

nest_asyncio.apply()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#################################################################
# 1) Preprocess the dataset for RAGAS
#################################################################
def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_input": row["Question"],            
            "retrieved_contexts": [row["Context"]],
            "response": row["Answer"],
            "reference": row["Ground_Truth"]
        })
    return EvaluationDataset.from_list(dataset)

#################################################################
# 2) Define the metrics
#################################################################
metrics = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    ResponseRelevancy(),
    Faithfulness(),
    FactualCorrectness(),
    NoiseSensitivity()
]

#################################################################
# 3) List of CSV files to process
#################################################################
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv",
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

#################################################################
# 4) List of unsloth model names
#################################################################
unsloth_models = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
]

#################################################################
# 5) Define a helper function for text generation
#################################################################
from ragas.llms import LangchainLLMWrapper

def load_and_build_llm(model_name: str):
    """
    Loads a Hugging Face model and tokenizer, returns a function
    that RAGAS can call for text generation.
    """
    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def llm_fn(prompt: str) -> str:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.inference_mode():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=256,
                temperature=0.7,
            )
        return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    return LangchainLLMWrapper(llm_fn)

#################################################################
# 6) Main evaluation loop (no callbacks used)
#################################################################
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"File not found: {csv_file}")
        continue

    dataset = preprocess_dataset(df)

    for model_name in unsloth_models:
        print(f"\nStarting evaluation for model: {model_name}")
        llm_fn = load_and_build_llm(model_name)

        try:
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm_fn,
                embeddings=None,  # Provide embeddings if needed
                raise_exceptions=True
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
            print(f"Evaluation failed for {model_name}: {e}")
            # Optionally inspect a few dataset entries for debugging
            for entry in dataset.to_pandas().head(3).to_dict(orient="records"):
                print("Debug entry:", entry)

    print(f"Finished processing dataset: {csv_file}")
