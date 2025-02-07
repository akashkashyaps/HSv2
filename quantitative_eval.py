import json
import re
from types import SimpleNamespace
from pydantic import BaseModel

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric
)
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
import pandas as pd
import torch
import nest_asyncio

nest_asyncio.apply()

# ----- HELPER FUNCTIONS -----
def fixup_required_keys(data: dict, schema: BaseModel) -> dict:
    required_fields = list(schema.model_fields.keys())
    
    if "verdicts" in required_fields:
        if not data.get("verdicts"):
            data["verdicts"] = [{"verdict": "idk"}]
        else:
            for verdict in data["verdicts"]:
                if "reason" not in verdict:
                    verdict["reason"] = ""  # Maintain required field structure

    if "statements" in required_fields:
        if not data.get("statements"):
            data["statements"] = ["idk"]

    return data

def parse_response(response_content: str, schema: BaseModel, debug: bool = True):
    try:
        parsed = json.loads(response_content)
    except json.JSONDecodeError:
        parsed = {}

    fixed = fixup_required_keys(parsed, schema)
    
    try:
        return schema(**fixed)
    except Exception as e:
        if debug:
            print(f"Validation error: {e}")
        return schema(**schema().dict())  # Fallback with schema defaults

# ----- MODEL WRAPPER -----
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name, debug: bool = True):
        self.model_name = model_name
        self.debug = debug
        self.model = ChatOllama(model=model_name, temperature=0, format="json")
        
    def load_model(self):
        return self.model
        
    def generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        response = self.model.invoke(prompt)
        return parse_response(response.content, schema, self.debug)
        
    async def a_generate(self, prompt: str, schema: BaseModel) -> BaseModel:
        response = await self.model.ainvoke(prompt)
        return parse_response(response.content, schema, self.debug)
        
    def get_model_name(self):
        return f"Ollama/{self.model_name}"
# ----- MAIN SCRIPT -----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# List of CSV files to process
csv_files = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv", 
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

def preprocess_dataset(df):
    test_cases = []
    for _, row in df.iterrows():
        test_cases.append(
            LLMTestCase(
                input=str(row["Question"]),
                actual_output=str(row["Answer"]),
                retrieval_context=[str(row["Context"])],
                expected_output=str(row["Ground_Truth"])
            )
        )
    return test_cases

# List of models to evaluate
models = [
    "mistral:7b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_0", 
    "qwen2.5:7b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0", 
    "phi3.5:3.8b-mini-instruct-q4_0",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:8b-llama-distill-q4_K_M",
    "lly/InternLM3-8B-Instruct:8b-instruct-q4_0"
]

def get_metrics(eval_model: DeepEvalBaseLLM):
    return [
        ContextualPrecisionMetric(
            threshold=0.7, 
            model=eval_model,
            strict_mode=True
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=eval_model,
            strict_mode=True
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=eval_model,
            strict_mode=True
        ),
        AnswerRelevancyMetric(
            threshold=0.75,
            model=eval_model,
            strict_mode=True
        ),
        ContextualRelevancyMetric(
            threshold=0.7,
            model=eval_model,
            strict_mode=True
        )
    ]

# ----- EVALUATION LOOP -----
for csv_file in csv_files:
    print(f"\nProcessing {csv_file}")
    df = pd.read_csv(csv_file)
    test_cases = preprocess_dataset(df)
    
    for model_name in models:
        print(f"\nEvaluating {model_name}")
        
        # Initialize model and metrics.
        ollama_model = OllamaModel(model_name)
        metrics = get_metrics(ollama_model)
        
        evaluation_result = evaluate(
            test_cases,
            metrics=metrics
        )
        
        # Collect results per test case with metrics.
        results = []
        for test_case_result in evaluation_result.results:
            result_data = {
                "model": model_name,
                "dataset": csv_file,
                "input": test_case_result.input,
                "actual_output": test_case_result.actual_output,
                "expected_output": test_case_result.expected_output
            }
            for metric in metrics:
                metric_name = metric.__class__.__name__
                result_data[f"{metric_name}_score"] = test_case_result.metric_scores.get(metric_name)
                result_data[f"{metric_name}_reason"] = test_case_result.metric_reasons.get(metric_name)
            results.append(result_data)
        
        results_df = pd.DataFrame(results)
        output_path = f"{csv_file.replace('.csv', '')}_DeepEval_{model_name}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Saved detailed results to {output_path}")
