import torch
import pandas as pd
import nest_asyncio
import json
import re
from typing import Dict, Any
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

class EnhancedJSONCallback(BaseCallbackHandler):
    def __init__(self):
        self.responses = []
        self.prompts = []
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts:
            self.prompts.append(prompts[0])
            print(f"Prompt: {prompts[0][:200]}...")  # Truncated for readability
    
    def on_llm_end(self, response, **kwargs):
        self.responses.append(response)
        try:
            parsed_response = json_fixer(response.generations[0][0].text)
            print(f"Parsed Response: {json.dumps(parsed_response, indent=2)}")
        except Exception as e:
            print(f"Failed to parse response: {str(e)}\nRaw response: {response}")

def json_fixer(llm_output: str) -> Dict[Any, Any]:
    """
    Enhanced JSON parser that handles malformed JSON responses.
    """
    # Remove any leading/trailing whitespace and common formatting issues
    cleaned_output = llm_output.strip()
    cleaned_output = re.sub(r'[\n\r\t]+', ' ', cleaned_output)
    
    try:
        # First attempt: direct JSON parsing
        return json.loads(cleaned_output)
    except json.JSONDecodeError:
        try:
            # Second attempt: find and parse the first JSON-like structure
            json_pattern = r'\{(?:[^{}]|(?R))*\}'
            matches = re.findall(json_pattern, cleaned_output, re.DOTALL)
            if matches:
                return json.loads(matches[0])
            
            # Third attempt: try to fix common JSON formatting issues
            fixed_output = cleaned_output
            fixed_output = re.sub(r'(?<!["{\[,])\b(true|false|null)\b(?!["}\],])', r'"\1"', fixed_output)
            fixed_output = re.sub(r'(?<=\w)"(?=\w)', '\\"', fixed_output)
            return json.loads(fixed_output)
        except Exception as e:
            # If all parsing attempts fail, return a structured error response
            return {
                "error": "Failed to parse JSON",
                "raw_output": cleaned_output,
                "error_details": str(e)
            }

def create_llm(model_name: str) -> ChatOllama:
    """
    Create a ChatOllama instance with improved JSON handling capabilities.
    """
    return ChatOllama(
        model=model_name,
        temperature=0,
        format="json",
        system=(
            "You are a JSON-only response system. Follow these rules:\n"
            "1. ALWAYS return valid JSON\n"
            "2. Use this exact schema: {\"response\": {\"content\": YOUR_CONTENT}}\n"
            "3. No markdown, no commentary, no extra text\n"
            "4. If you're unsure, return {\"error\": \"explanation\"}\n"
            "5. Always use double quotes for keys and string values"
        )
    )

def evaluate_model(model_name: str, dataset: EvaluationDataset, metrics: list) -> pd.DataFrame:
    """
    Evaluate a single model with enhanced error handling and logging.
    """
    llm = create_llm(model_name)
    callback = EnhancedJSONCallback()
    
    try:
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=llm,
            embeddings=OllamaEmbeddings(model="nomic-embed-text"),
            raise_exceptions=True,
            callbacks=[callback],
            run_config=RunConfig(
                timeout=60,
                max_retries=3,
                max_wait=60,
                max_workers=2
            )
        )
        return result.to_pandas()
    except Exception as e:
        print(f"Evaluation failed for {model_name}: {str(e)}")
        print("Last 3 prompts:", callback.prompts[-3:])
        print("Last 3 responses:", callback.responses[-3:])
        return pd.DataFrame({
            "model": [model_name],
            "error": [str(e)],
            "status": ["failed"]
        })