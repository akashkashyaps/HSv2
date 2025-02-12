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
from ragas.llms import LangchainLLMWrapper
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
        logging.FileHandler('ragas_evaluation.log'),
        logging.StreamHandler()
    ]
)

# Apply nest_asyncio for asynchronous support
nest_asyncio.apply()

class EnhancedCallback(BaseCallbackHandler):
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
                logging.debug(f"Raw response: {text}")
        except Exception as e:
            logging.error(f"Failed to log response: {str(e)}")

def create_wrapped_llm(model_name: str) -> LangchainLLMWrapper:
    """Create a LangchainLLMWrapper with ChatOllama."""
    base_llm = ChatOllama(
        model=model_name,
        temperature=0,
        system=(
            "You are an evaluation system for question-answering. Follow these rules:\n"
            "1. For binary decisions, respond with a clear 'yes' or 'no'\n"
            "2. For scoring, provide a number between 0 and 1\n"
            "3. Always maintain consistent response formats\n"
            "4. Focus on accuracy and relevance in evaluations"
        )
    )
    return LangchainLLMWrapper(llm=base_llm)

def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    """Prepare dataset for RAGAS evaluation."""
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: [x] if isinstance(x, str) else x
    )
    
    return EvaluationDataset.from_pandas(processed_df)

def evaluate_model(
    model_name: str,
    dataset: EvaluationDataset,
    metrics: List,
    output_dir: Path,
    csv_file: str
) -> Optional[pd.DataFrame]:
    """Evaluate a model with comprehensive error handling."""
    logging.info(f"Starting evaluation for model: {model_name}")
    
    try:
        wrapped_llm = create_wrapped_llm(model_name)
        callback = EnhancedCallback()
        callback.current_model = model_name
        
        # Run evaluation
        result = evaluate(
            dataset=dataset,
            metrics=metrics,
            llm=wrapped_llm,
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
        output_file = output_dir / f"{csv_file.replace('.csv', '')}_{model_name}_evaluation.csv"
        result_df.to_csv(output_file, index=False)
        
        logging.info(f"Successfully saved results to: {output_file}")
        return result_df
        
    except Exception as e:
        logging.error(f"Evaluation failed for {model_name}: {str(e)}")
        logging.debug("Last 3 prompts: %s", callback.prompts[-3:] if callback.prompts else "No prompts")
        logging.debug("Last 3 responses: %s", callback.responses[-3:] if callback.responses else "No responses")
        return None

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    # Configuration
    output_dir = Path("/home/akash/HSv2")
    output_dir.mkdir(exist_ok=True)
    
    # CSV files to process
    csv_files = [
        "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
        "Results_mistral:7b-instruct-q4_0.csv",
        "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
        "Results_gemma2:9b-instruct-q4_0.csv",
        "Results_qwen2.5:7b-instruct-q4_0.csv",
        "Results_llama3.1:8b-instruct-q4_0.csv"
    ]
    
    # Define models
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
    
    # Initialize wrapped LLM for metrics
    wrapped_base_llm = create_wrapped_llm(models[0])
    
    # Initialize metrics with wrapped LLM
    metrics = [
        LLMContextPrecisionWithReference(llm=wrapped_base_llm),
        LLMContextRecall(llm=wrapped_base_llm),
        ContextEntityRecall(llm=wrapped_base_llm),
        ResponseRelevancy(llm=wrapped_base_llm),
        Faithfulness(llm=wrapped_base_llm),
        FactualCorrectness(llm=wrapped_base_llm),
        NoiseSensitivity(llm=wrapped_base_llm)
    ]
    
    # Validate models
    logging.info("Running pre-flight model checks...")
    valid_models = []
    for model_name in models:
        try:
            test_llm = create_wrapped_llm(model_name)
            test_response = test_llm.invoke("Return JSON with key 'status' and value 'ok'")
            if test_response:
                valid_models.append(model_name)
                logging.info(f"✅ {model_name} passed test")
            else:
                logging.warning(f"❌ {model_name} failed: Invalid response")
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