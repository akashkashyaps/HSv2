import os
import re
import torch
import nest_asyncio
import pandas as pd

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.callbacks import BaseCallbackHandler
from ragas import evaluate, EvaluationDataset, RunConfig, RagasOutputParserException
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

###############################################
# 1) Callback to see prompts/responses (optional)
###############################################
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

###############################################
# 2) Split logic to keep only the first two Document(...) blocks
###############################################
def extract_first_two_documents(context_str: str) -> list[str]:
    """
    Naive approach: splits on 'Document(' and reconstructs only the first 2 Document(...) blocks.
    Returns them as separate strings so RAGAS interprets them as separate contexts.
    """
    if not context_str:
        return []
    # Split by 'Document('
    parts = context_str.split("Document(")
    # The first chunk, parts[0], is whatever came before the first Document( (often just '[')
    # Each subsequent chunk is the text after 'Document('
    # if we have fewer than 2 documents, just return the entire string as one context
    if len(parts) < 2:
        return [context_str]

    # Rebuild the first doc
    doc1 = "Document(" + parts[1]
    if len(parts) < 3:
        return [doc1]

    # Rebuild the second doc
    doc2 = "Document(" + parts[2]
    return [doc1, doc2]

###############################################
# 3) Preprocess function for RAGAS
###############################################
def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    rows = []
    for _, row in df.iterrows():
        # Keep only first 2 documents from the raw "Context" column
        raw_context = row["Context"] if "Context" in row else ""
        first_two_docs = extract_first_two_documents(str(raw_context))

        # Build the required RAGAS data structure
        rows.append({
            "user_input": row["Question"],              # The user question
            "retrieved_contexts": first_two_docs,       # At most 2 doc snippets
            "response": row["Answer"],                  # Model's generated response
            "reference": row["Ground_Truth"]            # Ground truth reference
        })
    return EvaluationDataset.from_list(rows)

###############################################
# 4) CSV files to process
###############################################
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv", 
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

###############################################
# 5) Models to evaluate
###############################################
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

###############################################
# 6) Metrics to evaluate
###############################################
metrics = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    ResponseRelevancy(),
    Faithfulness(),
    FactualCorrectness(),
    NoiseSensitivity()
]

###############################################
# 7) Main evaluation loop
###############################################
for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    if not os.path.isfile(csv_file):
        print(f"File not found: {csv_file}. Skipping.")
        continue

    df = pd.read_csv(csv_file)
    dataset = preprocess_dataset(df)

    for model_name in models:
        print(f"\nStarting evaluation for model: {model_name}")

        # Initialize ChatOllama with strict JSON output instructions
        llm = ChatOllama(
            model=model_name,
            temperature=0,
            format="json",
            system=(
                "You must return a valid JSON object only. "
                "Do not include any extra text, commentary, or formatting "
                "beyond the JSON object itself."
            )
        )
        ollama_emb = OllamaEmbeddings(model="nomic-embed-text")

        # Quick test query to see if the model adheres to JSON
        test_query = "Please return {\"check\":\"valid\"}"
        test_response = llm.invoke(test_query)
        print(f"Test response for model '{model_name}': {test_response}")

        try:
            # Run the evaluation with a callback for debug
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=ollama_emb,
                callbacks=[TestCallback()],
                # run_config with fewer retries/timeouts for demonstration
                run_config=RunConfig(
                    timeout=15,
                    max_retries=1,
                    max_wait=60,
                    max_workers=1
                ),
                raise_exceptions=True
            )
        except RagasOutputParserException as e:
            print(f"** JSON Parsing / RAGAS Output Error for model {model_name}:")
            print(e)
            # Print a snippet of the dataset for context
            for entry in dataset.to_pandas().head(2).to_dict(orient="records"):
                print("Debug entry:", entry)
            continue
        except Exception as e:
            print(f"** General Error for model {model_name}: {e}")
            continue

        # If evaluation succeeds, save to CSV
        output_file = f"/home/akash/HSv2/{csv_file.replace('.csv', '')}_Evaluator_{model_name}_quantitative.csv"
        result.to_pandas().to_csv(output_file, index=False)
        print(f"Completed evaluation for model: {model_name}")
        print(f"Results saved to: {output_file}")

    print(f"Finished processing dataset: {csv_file}")
