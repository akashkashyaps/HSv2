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

vectorstore = Chroma(persist_directory='/home/akash/HSv2/HSv2/', embedding_function=embeddings)

retriever = vectorstore.as_retriever(search_kwargs={'k': 1}, max_tokens=2500)


prompt_template =("""
<|user|>
For the purposes of this conversation, you are a helpful agent who is present at a University Open Day at Nottingham Trent University (NTU) at Clifton Campus, Nottingham, United Kingdom. A University Open Day is an event where future University students will visit to see the campus, facilities, and meet the teaching staff. The future students are going to ask you questions about University which you will answer by using the helpful context at the end of this message. Note that it is very important that you are at Nottingham Trent University (NTU) in the United Kingdom and NOT Nanyang Technological University in Singapore. You will now be given context and asked a question. Your task is to answer the question. If you do not know the answer, just say that you cannot answer the question, do not try to make up an answer.
CONTEXT: {context}
QUESTION: {question}<|end|>
<|assistant|>
Helpful Answer:
""")

prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

llm = HuggingFacePipeline(pipeline=generate_text)

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)   
query = "Are there any connections with employers?"
import time

start = time.time()
doc = retriever.get_relevant_documents(query)
chain.run(input_documents = doc, question = query)
end = time.time()

time_taken = end - start
time_taken
import re

# def process_chain_output(chain_output):
#     # Find the start index of the context
#     context_start = chain_output.find("don't try to make up an answer.\n") + len("don't try to make up an answer.\n")
    
#     # Extract relevant context
#     context_end = chain_output.find("Question:")
#     context = chain_output[context_start:context_end].strip()

#     # Extract helpful answer
#     answer_pattern = re.compile(r"Helpful Answer: (.+?)\n", re.DOTALL)
#     match = answer_pattern.search(chain_output)
#     if match:
#         answer = match.group(1).strip()
#     else:
#         answer = "I don't know."  # Set default answer if no helpful answer found

#     return answer, context

# # Test the function with the provided chain output
# chain_output = """Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# Careers in Computer Science

# The industry-focused nature of our courses ensures you stand out from the crowd when it comes to job applications and pursuing your future career. Our graduates are widely respected amongst employers and are perceived as having a competitive edge due to the hands-on approach of our teaching.

# ...

# Question: Are there any connections with local employers?
# Helpful Answer: I don't know.

# Answer: I don't know.

# == response ==

# While the provided information does not explicitly mention direct connections or partnerships between the university (NTU) and local employers, it is common for universities to establish relationships with nearby businesses for internships, collaborative projects, and recruitment opportunities. However, without specific details regarding such arrangements at NTU, we cannot confirm this aspect based solely on the given text."""

# answer, context = process_chain_output(chain_output)
# print("Answer:", answer)
# print("\nContext:", context)


# from datasets import Dataset
# from tqdm import tqdm
# import pandas as pd
# # Load test set
# test1000 = pd.read_csv('testset7-cleaned-JB-FFT.csv')
# questions = test1000['question'].tolist()
# questions
# # Create an empty list to store the results
# results = []

# # Loop through each question
# for question in questions:
#     doc = retriever.get_relevant_documents(question)
#     result = chain.run(input_documents=doc, question=question)
#     results.append(result)

# # Create a pandas DataFrame to store the results
# df = pd.DataFrame({"Question": questions, "Answer": results})
# df.to_csv('results4.csv', index=False)


import time
from datasets import Dataset
from tqdm import tqdm
import pandas as pd

# Load test set
test1000 = pd.read_csv('testset7-cleaned-JB-FFT.csv')
questions = test1000['question'].tolist()

# Create empty lists to store the results and the time taken
results = []
time_taken_list = []

# Loop through each question
for question in tqdm(questions):
    start = time.time()  # Start timing
    
    doc = retriever.get_relevant_documents(question)
    result = chain.run(input_documents=doc, question=question)
    
    end = time.time()  # End timing
    
    time_taken = end - start
    time_taken_list.append(time_taken)  # Store time taken
    
    results.append(result)

# Create a pandas DataFrame to store the results and time taken
df = pd.DataFrame({"Question": questions, "Answer": results, "Time_Taken": time_taken_list})
df.to_csv('Phi_time.csv', index=False)


# # Function to generate answers using RAG and retrieve context using vector store
# def generate_answer(question):
#     doc = vectorstore.similarity_search(question)
#     answer = chain.run(input_documents=doc, question=question)
#     return answer

# # Create new columns for answers and context
# test100['answer'] = ""
# test100['context'] = ""

# # Loop through each question
# for index, row in tqdm(test100.iterrows(), total=len(test100)):
#     question = row['question']
    
#     # Generate answer and retrieve context
#     answer = generate_answer(question)
#     context = doc  # Assuming you want to store the retrieved context
    
#     # Store the answer and context in the DataFrame
#     test100.at[index, 'answer'] = answer
#     test100.at[index, 'context'] = context

# test100.to_csv('processed_data2.csv', index=False)

# # Prepare dataset for batch processing
# dataset = Dataset.from_pandas(test100)

# # Define the process_chain_output function
# def process_chain_output(chain_output):
#     # Find the start index of the context
#     context_start = chain_output.find("don't try to make up an answer.\n") + len("don't try to make up an answer.\n")
    
#     # Extract relevant context
#     context_end = chain_output.find("Question:")
#     context = chain_output[context_start:context_end].strip()

#     # Extract helpful answer
#     answer_pattern = re.compile(r"Helpful Answer: (.+?)(\n/)?\n", re.DOTALL)
#     match = answer_pattern.search(chain_output)
#     if match:
#         answer = match.group(1).strip()
#     else:
#         answer = "I don't know."  # Set default answer if no helpful answer found

#     return answer, context

# # Define the process_question function
# def process_question(question):
#     # Run the chain on the question
#     doc = vectorstore.similarity_search(question)
#     chain_output = chain.run(input_documents=doc, question=question)
#     # Process the chain output to extract answer and context
#     answer, context = process_chain_output(chain_output)
#     return answer, context

# # Create empty lists to store answers and contexts
# answers = []
# contexts = []

# # Iterate over each question in the DataFrame
# for index, row in test100.iterrows():
#     question = row['question']  # Replace 'your_question_column_name' with the actual column name containing the questions
#     # Process the question
#     answer, context = process_question(question)
#     answers.append(answer)
#     contexts.append(context)

# # Add the answers and contexts lists to the DataFrame
# test100['answer'] = answers
# test100['context'] = contexts

# test100.to_csv('processed_data4.csv', index=False)

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

# def create_ragas_dataset(chain, dataset, vectorstore):
#     rag_dataset = []
#     for idx, row in tqdm(dataset.iterrows(), total=len(dataset)):
#         # Run the chain to get the answer
#         answer = chain.run(input_documents=vectorstore.similarity_search(row["question"]), question=row["question"])
        
#         # Generating context using vectorstore
#         context_str = vectorstore.similarity_search(row["question"])
#         context_list = context_str.split('\n')  # Split by newline character
        
#         # Append the result to the dataset list
#         rag_dataset.append(
#             {
#                 "question": row["question"],
#                 "answer": answer,
#                 "context": context_list,  # Store context as a list
#                 "contexts": row["contexts"],
#                 "ground_truth": row["ground_truth"],
#             }
#         )
    
#     # Convert list to DataFrame
#     rag_df = pd.DataFrame(rag_dataset)
    
#     # Convert DataFrame to Dataset
#     rag_eval_dataset = Dataset.from_pandas(rag_df)
    
#     return rag_eval_dataset


# def evaluate_ragas_dataset(ragas_dataset):
#   result = evaluate(
#     ragas_dataset,
#     llm=llm,
#     embeddings=embeddings,
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

# from tqdm import tqdm
# import pandas as pd

# qa_ragas_dataset4 = create_ragas_dataset(chain, dataset, vectorstore)
# qa_ragas_dataset4[0]

# qa_ragas_dataset4.to_csv('qa_ragas_dataset4.csv', index=False)

# qa_result = evaluate_ragas_dataset(qa_ragas_dataset4)
# qa_result[0]

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

