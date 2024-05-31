# Import dependancies
import torch
import transformers
import pandas as pd
import numpy as np
from torch import cuda, bfloat16
from langchain.llms import HuggingFacePipeline
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain.chains import RetrievalQA
from transformers import StoppingCriteriaList, StoppingCriteria
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub
from langchain.prompts import PromptTemplate

# Define the model
model_id = "microsoft/Phi-3-mini-4k-instruct"
hf_auth = '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'
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    trust_remote_code=True
)
device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'

# Configure BitsAndBytesConfig for quantization
bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

model_config = transformers.AutoConfig.from_pretrained(
    model_id,
    use_auth_token=hf_auth,
    trust_remote_code=True
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    use_auth_token=hf_auth,
)   

# Set the model to evaluation mode
model.eval()

# Define the text generation pipeline
generate_text = transformers.pipeline(
    model=model,
    tokenizer=tokenizer,
    return_full_text=True,
    task='text-generation',
    max_new_tokens=512,
    do_sample=True,
    temperature=0.1,
    repetition_penalty=1.1
)

# # Test the model with a prompt
# input_text = "what features does nottingham trent university computer science department have?"
# generated_ids = generate_text(input_text)[0]['generated_text']
# print(generated_ids)


# from langchain.document_loaders import SeleniumURLLoader

# urls = ["https://www.ntu.ac.uk/course/computer-science", 
#         "https://www.ntu.ac.uk/study-and-courses/academic-schools/science-and-technology/Computer-Science"]
       
# loader = SeleniumURLLoader(urls)
# documents = loader.load()

# # Split the documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
# all_splits = text_splitter.split_documents(documents)

# Define the model name and kwargs for embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# Create the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# # Create the vectorstore
# vectorstore = Chroma.from_documents(
#     all_splits,
#     embeddings,
#     persist_directory="/home/akash/HSv2"
# )

vectorstore = Chroma(persist_directory='/home/akash/HSv2/', embedding_function=embeddings)

retriever = vectorstore.as_retriever()


prompt_template =("""
<|user|>
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
NTU is Nottingham Trent University.It is not Singapore. Note this it is very important.
{context}
Question: {query}<|end|>
<|assistant|>
Helpful Answer:
""")

prompt=PromptTemplate(template=prompt_template,input_variables=["context","query"])

llm = HuggingFacePipeline(pipeline=generate_text)

chain = load_qa_chain(llm, chain_type="stuff")
query = "Are there any connections with local employers?"
doc = vectorstore.similarity_search(query)
chain.run(input_documents = doc, question = query)


from datasets import Dataset

# Load test set
test100 = pd.read_csv('100-phi3-llama3.csv')

# Prepare dataset for batch processing
dataset = Dataset.from_pandas(test100)

from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    context_relevancy,
    answer_correctness,
    answer_similarity
)

from ragas.metrics.critique import harmfulness
from ragas import evaluate

def create_ragas_dataset(chain, dataset, vectorstore):
    rag_dataset = []
    for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
        # Run the chain to get the answer and context
        answer = chain.run(input_documents=vectorstore.similarity_search(row["question"]), question=row["question"])
        context = vectorstore.similarity_search(row["question"])  # Generating context using vectorstore
        
        # Append the result to the dataset list
        rag_dataset.append(
            {
                "question": row["question"],
                "answer": answer,
                "context": context,
                "contexts": row["contexts"],
                "ground_truth": row["ground_truth"],
            }
        )
    
    # Convert list to DataFrame
    rag_df = pd.DataFrame(rag_dataset)
    
    # Convert DataFrame to Dataset
    rag_eval_dataset = Dataset.from_pandas(rag_df)
    
    return rag_eval_dataset



def evaluate_ragas_dataset(ragas_dataset):
  result = evaluate(
    ragas_dataset,
    llm=llm,
    embeddings=embeddings,
    metrics=[
        context_precision,
        faithfulness,
        answer_relevancy,
        context_recall,
        context_relevancy,
        answer_correctness,
        answer_similarity
    ],
  )
  return result

from tqdm import tqdm
import pandas as pd

qa_ragas_dataset4 = create_ragas_dataset(chain, dataset, vectorstore)
qa_ragas_dataset4[0]

qa_ragas_dataset4.to_csv('qa_ragas_dataset4.csv', index=False)

qa_result = evaluate_ragas_dataset(qa_ragas_dataset4)
qa_result[0]

# # Function to process each question
# def process_batch(batch):
#     questions = batch["question"]
#     contexts = []
#     answers = []
    
#     for question in questions:
#         docs = vectorstore.similarity_search(question)
#         retrieved_context = " ".join([doc.page_content for doc in docs])
#         result = chain.run(input_documents=docs, question=question)
        
#         if '\nHelpful Answer:' in result:
#             context, answer = result.split('\nHelpful Answer:', 1)
#             context = context.strip()
#             answer = answer.strip()
#         else:
#             context = retrieved_context
#             answer = "I don't know"
        
#         contexts.append(context)
#         answers.append(answer)
    
#     return {"RAG_context": contexts, "answer": answers}

# # # Batch process the dataset
# # results = dataset.map(process_batch, batched=True, batch_size=8)
# import dill
# # Serialize the process_batch function using dill
# serialized_function = dill.dumps(process_batch)

# # Define a wrapper function to deserialize and call the process_batch function
# def process_batch_wrapper(batch):
#     func = dill.loads(serialized_function)
#     return func(batch)

# # Batch process the dataset
# results = dataset.map(process_batch_wrapper, batched=True, batch_size=8)

# def generate_answer(question, vectorstore, chain):
#     docs = vectorstore.similarity_search(question)
#     retrieved_context = " ".join([doc.page_content for doc in docs])
#     result = chain.run(input_documents=docs, question=question)

#     if '\nHelpful Answer:' in result:
#         context, answer = result.split('\nHelpful Answer:', 1)
#         context = context.strip()
#         answer = answer.strip()(chain.run(input_documents = doc, question = "question"), dataset)
#     else:
#         context = retrieved_context
#         answer = "I don't know"

#     return context, answer

# def process_batch(batch, vectorstore, chain):
#     questions = batch["question"]
#     contexts = []
#     answers = []

#     for question in questions:
#         context, answer = generate_answer(question, vectorstore, chain)
#         contexts.append(context)
#         answers.append(answer)

#     return {"RAG_context": contexts, "answer": answers}

# results = dataset.map(lambda batch: process_batch(batch, vectorstore, chain), batched=True, batch_size=8)


# # Add the results to the DataFrame
# test100['RAG_context'] = results['RAG_context']
# test100['answer'] = results['answer']

# # Save the DataFrame with the results
# test100.to_csv('test100_with_results.csv', index=False)

# from ragas.metrics import (
#     answer_relevancy,
#     faithfulness,
#     context_recall,
#     context_precision,
#     context_relevancy,
#     answer_correctness,
#     answer_similarity
# )

# from ragas.metrics.critique import harmfulness
# from ragas import evaluate
# from tqdm import tqdm

# def create_ragas_dataset(rag_pipeline, eval_dataset):
#   rag_dataset = []
#   for row in tqdm(eval_dataset):
#     answer = rag_pipeline.invoke({"question" : row["question"]})
#     rag_dataset.append(
#         {"question" : row["question"],
#          "answer" : answer["response"],
#          "contexts" : [context.page_content for context in answer["context"]],
#          "ground_truths" : [row["ground_truth"]]
#          }
#     )
#   rag_df = pd.DataFrame(rag_dataset)
#   rag_eval_dataset = Dataset.from_pandas(rag_df)
#   return rag_eval_dataset

# def evaluate_ragas_dataset(ragas_dataset):
#   result = evaluate(
#     ragas_dataset,
#     metrics=[
#         context_precision,
#         faithfulness,
#         answer_relevancy,
#         context_recall,
#         context_relevancy,
#         answer_correctness,
#         answer_similarity
#     ],
#   )
#   return result

# df = result.to_pandas()
# df.head()

# finaltest1 = pd.read_csv('testset3_with_results.csv')
# # finaltest1.drop(columns=['contexts'], inplace=True)
# finaltest1.rename(columns={'newAnswer': 'answer'}, inplace=True)
# finaltest1


# dataset = Dataset.from_pandas(finaltest1)
# print(dataset.features["contexts"])

# import ast

# def preprocess_contexts(example):
#     example["contexts"] = ast.literal_eval(example["contexts"])
#     return example

# dataset = dataset.map(preprocess_contexts)


# score = evaluate(
#     dataset,
#     llm=llm,
#     embeddings=embeddings,
#     metrics=[
#         context_precision,
#         faithfulness,
#         answer_relevancy,
#         context_recall,
#     ],
# )

# from ragas import evaluate

# result = evaluate(
#     finaltest1,
#     metrics=[
#         context_precision,
#         faithfulness,
#         answer_relevancy,
#         context_recall,
#     ],
# )

# result

