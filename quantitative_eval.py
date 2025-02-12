import torch
import pandas as pd
import nest_asyncio
import json
import re
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

# Apply nest_asyncio for asynchronous support
nest_asyncio.apply()

class EnhancedJSONCallback(BaseCallbackHandler):
    """Enhanced callback handler with JSON validation and logging."""
    
    def __init__(self):
        self.responses = []
        self.prompts = []
        self.current_model = None
        
    def on_llm_start(self, serialized, prompts, **kwargs):
        if prompts:
            self.prompts.append(prompts[0])
            logging.debug(f"Processing prompt: {prompts[0][:200]}...")
    
    def on_llm_end(self, response, **kwargs):
        self.responses.append(response)
        try:
            if hasattr(response, 'generations') and response.generations:
                text = response.generations[0][0].text
                parsed_response = json_fixer(text)
                logging.debug(f"Successfully parsed response: {json.dumps(parsed_response, indent=2)}")
            else:
                logging.warning("Response object has unexpected structure")
        except Exception as e:
            logging.error(f"Failed to parse response: {str(e)}")
            logging.debug(f"Raw response: {response}")

def json_fixer(llm_output: str) -> Dict[Any, Any]:
    """
    Advanced JSON parser with multiple fallback strategies.
    """
    if not isinstance(llm_output, str):
        return {"error": f"Input is not a string: {type(llm_output)}"}
    
    # Clean the input
    cleaned_output = llm_output.strip()
    cleaned_output = re.sub(r'[\n\r\t]+', ' ', cleaned_output)
    
    # Multiple parsing attempts
    parsing_attempts = [
        # Attempt 1: Direct parsing
        lambda x: json.loads(x),
        
        # Attempt 2: Fix common JSON formatting issues
        lambda x: json.loads(re.sub(r'(?<!["{\[,])\b(true|false|null)\b(?!["}\],])', r'"\1"', x)),
        
        # Attempt 3: Extract JSON-like structure
        lambda x: json.loads(re.findall(r'\{(?:[^{}]|(?R))*\}', x, re.DOTALL)[0]),
        
        # Attempt 4: Handle single quotes
        lambda x: json.loads(x.replace("'", '"')),
        
        # Attempt 5: Fix unescaped quotes
        lambda x: json.loads(re.sub(r'(?<=\w)"(?=\w)', r'\"', x))
    ]
    
    for attempt_func in parsing_attempts:
        try:
            return attempt_func(cleaned_output)
        except Exception:
            continue
    
    # If all attempts fail, create a structured error response
    return {
        "error": "Failed to parse JSON",
        "raw_output": cleaned_output[:500],  # Truncate very long outputs
        "timestamp": datetime.now().isoformat()
    }

def create_llm(model_name: str) -> ChatOllama:
    """
    Create a ChatOllama instance with strict JSON formatting requirements.
    """
    return ChatOllama(
        model=model_name,
        temperature=0,
        format="json",
        system=(
            "You are a JSON-only response system. Rules:\n"
            "1. ALWAYS return valid JSON\n"
            "2. Use this schema: {\"response\": {\"content\": YOUR_CONTENT}}\n"
            "3. No markdown, no commentary, no extra text\n"
            "4. Use double quotes for ALL keys and string values\n"
            "5. If uncertain, return {\"error\": \"explanation\"}\n"
            "6. Escape special characters properly\n"
            "7. Arrays must use square brackets []\n"
            "8. Numbers should not be quoted\n"
        )
    )

def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    """
    Prepare dataset for RAGAS evaluation.
    """
    return EvaluationDataset.from_pandas(
        df.rename(columns={
            "Question": "user_input",
            "Context": "retrieved_contexts",
            "Answer": "response",
            "Ground_Truth": "reference"
        }).assign(
            retrieved_contexts=lambda x: x.retrieved_contexts.apply(lambda y: [y])
        )
    )

def evaluate_model(
    model_name: str,
    dataset: EvaluationDataset,
    metrics: List,
    output_dir: Path,
    csv_file: str
) -> Optional[pd.DataFrame]:
    """
    Evaluate a single model with comprehensive error handling and logging.
    """
    logging.info(f"Starting evaluation for model: {model_name}")
    
    llm = create_llm(model_name)
    callback = EnhancedJSONCallback()
    callback.current_model = model_name
    
    try:
        # Test JSON capability
        test_response = llm.invoke("Return a simple JSON response")
        _ = json_fixer(test_response.content)
        
        # Run evaluation
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
        
        # Save results
        result_df = result.to_pandas()
        output_file = output_dir / f"{csv_file.replace('.csv', '')}_{model_name}_quantitative.csv"
        result_df.to_csv(output_file, index=False)
        
        logging.info(f"Successfully saved results to: {output_file}")
        return result_df
        
    except Exception as e:
        logging.error(f"Evaluation failed for {model_name}: {str(e)}")
        logging.debug("Last 3 prompts: %s", callback.prompts[-3:] if callback.prompts else "No prompts")
        logging.debug("Last 3 responses: %s", callback.responses[-3:] if callback.responses else "No responses")
        
        # Save error information
        error_df = pd.DataFrame({
            "model": [model_name],
            "error": [str(e)],
            "status": ["failed"],
            "timestamp": [datetime.now().isoformat()]
        })
        error_file = output_dir / f"{csv_file.replace('.csv', '')}_{model_name}_error.csv"
        error_df.to_csv(error_file, index=False)
        return None

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    output_dir = Path("/home/akash/HSv2")
    output_dir.mkdir(exist_ok=True)
    
    csv_files = [
        "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
        "Results_mistral:7b-instruct-q4_0.csv",
        "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
        "Results_gemma2:9b-instruct-q4_0.csv",
        "Results_qwen2.5:7b-instruct-q4_0.csv",
        "Results_llama3.1:8b-instruct-q4_0.csv"
    ]
    
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
    
    metrics = [
        LLMContextPrecisionWithReference(),
        LLMContextRecall(),
        ContextEntityRecall(),
        ResponseRelevancy(),
        Faithfulness(),
        FactualCorrectness(),
        NoiseSensitivity()
    ]
    
    # Validate models
    logging.info("Running pre-flight model checks...")
    valid_models = []
    for model_name in models:
        try:
            test_llm = ChatOllama(model=model_name)
            response = test_llm.invoke("Return JSON with key 'status' and value 'ok'")
            parsed = json_fixer(response.content)
            if isinstance(parsed, dict):
                valid_models.append(model_name)
                logging.info(f"✅ {model_name} passed JSON test")
            else:
                logging.warning(f"❌ {model_name} failed: Invalid JSON response")
        except Exception as e:
            logging.error(f"❌ {model_name} failed: {str(e)}")
    
    # Main evaluation loop
    for csv_file in csv_files:
        logging.info(f"\nProcessing dataset: {csv_file}")
        try:
            df = pd.read_csv(csv_file)
            dataset = preprocess_dataset(df)
            
            for model_name in valid_models:
                result_df = evaluate_model(
                    model_name=model_name,
                    dataset=dataset,
                    metrics=metrics,
                    output_dir=output_dir,
                    csv_file=csv_file
                )
                
                if result_df is not None:
                    logging.info(f"Successfully evaluated {model_name} on {csv_file}")
                else:
                    logging.warning(f"Evaluation failed for {model_name} on {csv_file}")
                    
        except Exception as e:
            logging.error(f"Failed to process {csv_file}: {str(e)}")
            continue

if __name__ == "__main__":
    main()