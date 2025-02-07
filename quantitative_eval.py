from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    AnswerRelevancyMetric,
    ContextualRelevancyMetric,
    HallucinationMetric
)
from deepeval.models import DeepEvalBaseLLM
from langchain_ollama import ChatOllama
import pandas as pd
import torch
import nest_asyncio

nest_asyncio.apply()

# Improved Ollama model wrapper with proper async support
class OllamaModel(DeepEvalBaseLLM):
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = ChatOllama(model=model_name, temperature=0.6)
        
    def load_model(self):
        return self.model
        
    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content
        
    async def a_generate(self, prompt: str) -> str:
        response = await self.model.ainvoke(prompt)
        return response.content
        
    def get_model_name(self):
        return f"Ollama/{self.model_name}"  # Use stored model name

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

# Preprocess the dataset to match deepeval expected format
def preprocess_dataset(df):
    test_cases = []
    for _, row in df.iterrows():
        test_cases.append(LLMTestCase(
            input=str(row["Question"]),
            actual_output=str(row["Answer"]),
            retrieval_context=[str(row["Context"])],
            expected_output=str(row["Ground_Truth"])
        ))
    return test_cases

# List of models to evaluate
models = ["llama3.1:8b-instruct-q4_0", "qwen2.5:7b-instruct-q4_0", "gemma2:9b-instruct-q4_0", "phi3.5:3.8b-mini-instruct-q4_0", "mistral:7b-instruct-q4_0","deepseek-r1:7b-qwen-distill-q4_K_M","deepseek-r1:8b-llama-distill-q4_K_M","lly/InternLM3-8B-Instruct:8b-instruct-q4_0"]  

# Define the metrics to evaluate
def get_metrics(eval_model: DeepEvalBaseLLM):
    return [
        ContextualPrecisionMetric(
            threshold=0.7, 
            model=eval_model,  # Use Ollama model instead of GPT
            include_reason=True
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True
        ),
        AnswerRelevancyMetric(
            threshold=0.75,
            model=eval_model,
            include_reason=True
        ),
        HallucinationMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True
        ),
        ContextualRelevancyMetric(
            threshold=0.7,
            model=eval_model,
            include_reason=True
        )
    ]

# Modified evaluation loop
for csv_file in csv_files:
    print(f"\nProcessing {csv_file}")
    df = pd.read_csv(csv_file)
    test_cases = preprocess_dataset(df)
    
    for model_name in models:
        print(f"Evaluating {model_name}")
        
        # Initialize model and metrics
        ollama_model = OllamaModel(model_name)
        metrics = get_metrics(ollama_model)  # Create metrics with current model
        
        evaluation_result = evaluate(
            test_cases,
            metrics=metrics,
            model=ollama_model,
            show_progress_bar=True
        )
        
        # Collect results per test case with metrics
        results = []
        for test_case_result in evaluation_result.results:
            result_data = {
                "model": model_name,
                "dataset": csv_file,
                "input": test_case_result.input,
                "actual_output": test_case_result.actual_output,
                "expected_output": test_case_result.expected_output
            }
            
            # Add metric scores and reasons
            for metric in metrics:
                metric_name = metric.__class__.__name__
                result_data[f"{metric_name}_score"] = test_case_result.metric_scores[metric_name]
                result_data[f"{metric_name}_reason"] = test_case_result.metric_reasons[metric_name]
            
            results.append(result_data)
        
        # Save results with improved formatting
        results_df = pd.DataFrame(results)
        output_path = f"{csv_file.replace('.csv', '')}_DeepEval_{model_name}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Saved detailed results to {output_path}")
