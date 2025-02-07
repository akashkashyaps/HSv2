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

# ----- SCHEMA DEFINITIONS -----
# (Make sure these match your actual schemas or adjust as needed.)
class Statements(BaseModel):
    statements: list[str]


class AnswerRelevancyVerdict(BaseModel):
    verdict: str  # Expected to be either "yes", "no", or "idk"
    reason: str | None = None


class Verdicts(BaseModel):
    verdicts: list[AnswerRelevancyVerdict]


class Reason(BaseModel):
    reason: str

# ----- HELPER FUNCTIONS -----
def fixup_required_keys(data: dict, schema: BaseModel) -> dict:
    """
    For a given schema, if the required key(s) are missing or empty,
    populate them with a default value.
    This function is general and you can adjust the default values as needed.
    """
    if schema.__name__ == "Statements":
        if "statements" not in data or not data.get("statements"):
            data["statements"] = ["idk"]
    elif schema.__name__ == "Verdicts":
        if "verdicts" not in data or not data.get("verdicts"):
            data["verdicts"] = [{"verdict": "idk", "reason": None}]
    elif schema.__name__ == "Reason":
        if "reason" not in data or not data.get("reason"):
            data["reason"] = "idk"
    # For a general schema, you could loop over schema.__fields__
    # and add defaults if desired.
    return data

def parse_response(response_content: str, schema: BaseModel = None, debug: bool = True):
    """
    Parse the LLM response content.
    
    1. Try to decode the JSON.
    2. If a schema is provided, check whether any of its required keys are present.
       (This is determined by looking at the schema’s field names.)
       – If none are present, ignore the returned JSON (set data = {}).
    3. “Fix up” the parsed data so that missing or empty required keys are populated with default values.
    4. Instantiate the schema (if provided) or return a SimpleNamespace.
    """
    if debug:
        print("DEBUG: Raw response content:")
        print(response_content)
    try:
        parsed = json.loads(response_content)
        if debug:
            print("DEBUG: Parsed JSON:")
            print(parsed)
    except Exception as e:
        if debug:
            print("DEBUG: JSON parsing failed:", e)
        parsed = {}

    if schema is not None:
        # Determine the set of required keys from the schema.
        required_keys = list(schema.__fields__.keys())
        # Check if any required key is found
        found = any(key in parsed for key in required_keys)
        if not found:
            if debug:
                print(f"DEBUG: None of the required keys {required_keys} found. "
                      f"Ignoring returned JSON and using default values.")
            parsed = {}

        fixed = fixup_required_keys(parsed, schema)
        if debug:
            print("DEBUG: Fixed parsed JSON:")
            print(fixed)
        try:
            return schema(**fixed)
        except Exception as e:
            raise ValueError(f"Error instantiating schema with data: {fixed}\nError: {e}")
    else:
        return SimpleNamespace(**parsed)

# ----- MODEL WRAPPER -----
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name, debug: bool = True):
        self.model_name = model_name
        self.debug = debug
        # Force the LLM to return JSON.
        self.model = ChatOllama(model=model_name, temperature=0, format="json")
        
    def load_model(self):
        return self.model
        
    def generate(self, prompt: str, **kwargs):
        """
        Synchronously call the LLM and parse its response.
        Optionally pass a "schema" via kwargs for automatic validation and fixup.
        """
        response = self.model.invoke(prompt)
        schema = kwargs.get("schema", None)
        return parse_response(response.content, schema, debug=self.debug)
        
    async def a_generate(self, prompt: str, **kwargs):
        """
        Asynchronously call the LLM and parse its response.
        Optionally pass a "schema" via kwargs.
        """
        response = await self.model.ainvoke(prompt)
        schema = kwargs.get("schema", None)
        return parse_response(response.content, schema, debug=self.debug)
        
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
        ollama_model = OllamaModel(model_name, debug=True)
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
