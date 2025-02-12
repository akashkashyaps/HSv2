import pandas as pd
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
from langchain import HuggingFacePipeline
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_utilization
)
from ragas import evaluate
from datasets import Dataset

import nest_asyncio
nest_asyncio.apply()

# embedding model
embedding_model = SentenceTransformer("microsoft/mpnet-base")

# evaluator
model_id = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id)

device = 0  # Use GPU (0 is typically the first GPU device)

pipe = pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,  # langchain expects the full text
    task='text-generation',
    temperature=0.1,
    do_sample=True,
    max_new_tokens = 200,
    repetition_penalty=1.1  # without this output begins repeating

)

evaluator = HuggingFacePipeline(pipeline=pipe)

data_samples = {
    'question': ['When was the first super bowl?', 'Who won the most super bowls?'],
    'answer': ['The first superbowl was held on Jan 15, 1967', 'The most super bowls have been won by The New England Patriots'],
    'contexts' : [['The First AFL–NFL World Championship Game was an American football game played on January 15, 1967, at the Los Angeles Memorial Coliseum in Los Angeles,'], 
    ['The Green Bay Packers...Green Bay, Wisconsin.','The Packers compete...Football Conference']],
}
dataset = Dataset.from_dict(data_samples)

# ragas
result = evaluate(
    dataset=dataset,
    llm=evaluator,
    embeddings=embedding_model,
    raise_exceptions=False,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_utilization,
    ]
)

print(result)