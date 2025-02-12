import torch
import pandas as pd
import nest_asyncio
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from ragas import evaluate, EvaluationDataset
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

# Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Updated models list with Hugging Face models
models = [
    "unsloth/Meta-Llama-3.1-8B-Instruct-unsloth-bnb-4bit",
    "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "unsloth/mistral-7b-instruct-v0.3-bnb-4bit",
    "unsloth/gemma-2-9b-it-bnb-4bit",
    "unsloth/Phi-3.5-mini-instruct-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit",
    "unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit"
]

# CSV files should be updated to match model names
csv_files = [f"Results_{model.replace('/', '_').replace('-', '_')}.csv" for model in models]

def preprocess_dataset(df: pd.DataFrame) -> EvaluationDataset:
    # Same preprocessing as before
    dataset = []
    for _, row in df.iterrows():
        dataset.append({
            "user_input": row["Question"],
            "retrieved_contexts": [row["Context"]],
            "response": row["Answer"],
            "reference": row["Ground_Truth"]
        })
    return EvaluationDataset.from_list(dataset)

def create_pipeline(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        trust_remote_code=True
    )
    
    return pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=device,
        max_new_tokens=512,
        do_sample=False,
        return_full_text=False
    )

metrics = [
    LLMContextPrecisionWithReference(),
    LLMContextRecall(),
    ContextEntityRecall(),
    ResponseRelevancy(),
    Faithfulness(),
    FactualCorrectness(),
    NoiseSensitivity()
]

for csv_file in csv_files:
    print(f"\nProcessing dataset: {csv_file}")
    df = pd.read_csv(csv_file)
    dataset = preprocess_dataset(df)

    for model_name in models:
        print(f"\nStarting evaluation for model: {model_name}")

        try:
            # Create Hugging Face pipeline
            pipe = create_pipeline(model_name)
            llm = HuggingFacePipeline(pipeline=pipe)
            
            # Use all-MiniLM-L6-v2 for embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": device}
            )

            # Test the model
            test_query = "Please return a valid JSON object with a single key 'result' and a simple value."
            test_response = llm.invoke(test_query)
            print(f"Test response: {test_response}")

            # Run evaluation
            result = evaluate(
                dataset=dataset,
                metrics=metrics,
                llm=llm,
                embeddings=embeddings,
                raise_exceptions=True
            )

            # Save results
            output_file = f"/path/to/results/{model_name.replace('/', '_')}_evaluation.csv"
            result.to_pandas().to_csv(output_file, index=False)
            print(f"Evaluation completed for {model_name}")

        except Exception as e:
            print(f"Error evaluating {model_name}: {str(e)}")
            continue

print("All evaluations completed!")