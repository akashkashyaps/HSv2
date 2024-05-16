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

# Define the model
model_id = "google/gemma-2b"
hf_auth = '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'
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    use_auth_token=hf_auth
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
    use_auth_token=hf_auth
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

# Create the HuggingFacePipeline
llm = HuggingFacePipeline(pipeline=generate_text)

# Load the CSV files using CSVLoader
loaders = [
    CSVLoader(file_path='CCBI_cleaned.csv'),
    CSVLoader(file_path='CD_dept_cleaned.csv'),
    CSVLoader(file_path='CS_facilities_cleaned.csv'),
    CSVLoader(file_path='CS_PG_cleaned.csv'),
    CSVLoader(file_path='CS_UG_cleaned.csv'),
    CSVLoader(file_path='CSRG_cleaned.csv'),
    CSVLoader(file_path='ISRG_cleaned.csv'),
    CSVLoader(file_path='ISTeC_cleaned.csv'),
    CSVLoader(file_path='CSHomePage_cleaned.csv')
]

# Load the documents
documents = []
for loader in loaders:
    documents.extend(loader.load())

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Define the model name and kwargs for embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# Create the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Create the vector store using ChromaDBYour script looks well-structured but there are a few improvements and clarifications that can be made to ensure it runs correctly. Here’s the enhanced version of your script with added explanations:
vectorstore = Chroma.from_documents(
    all_splits,
    embeddings,
    persist_directory="/home/akash/historichatv1"
)

retriever = vectorstore.as_retriever()
import os
os.environ['HUGGINGFACEHUB_API_TOKEN']="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"

from langchain_community.llms import HuggingFaceHub

hf=HuggingFaceHub(
    repo_id="google/gemma-2b",
    model_kwargs={"temperature":0.1,"max_length":500}

)
query="What facilities are there in the CS department in Nottingham Trent University?"
hf.invoke(query)

input_text = "What facilities are there in the CS department in Nottingham Trent University?"
input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs[0]))

from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")

input_text = "What facilities are there in the CS department in Nottingham Trent University?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids)
print(tokenizer.decode(outputs))

prompt_template =("""
<start_of_turn>user
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

{context}

Question: {query}<end_of_turn>
<start_of_turn>model
Helpful Answer:
""")
from langchain.prompts import PromptTemplate
prompt=PromptTemplate(template=prompt_template,input_variables=["context","query"])

retrievalQA=RetrievalQA.from_chain_type(
    llm=hf,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
    # chain_type_kwargs={"prompt":prompt}
)

query="What facilities are there in the CS department in Nottingham Trent University?"
# Call the QA chain with our query.
query = "What facilities are there in the CS department in Nottingham Trent University?"
input_dict = {"query": query}
print(input_dict)
result = retrievalQA.invoke(input_dict)
print(result['result'])


# #vector search
# query = "What is CCBI?"
# result = vectorstore.similarity_search(query)
# print(result)

# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_template("""
# <start_of_turn>user
# Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {query}<end_of_turn>
# <start_of_turn>model
# """)

# from langchain.chains.combine_documents import create_stuff_documents_chain

# # Create the question-answering chain
# question_answer_chain = create_stuff_documents_chain(llm, prompt)
# from langchain.chains import ConversationalRetrievalChain, create_retrieval_chain, create_stuff_documents_chain
# document_chain = create_stuff_documents_chain(llm,prompt)
# retriever = vectorstore.as_retriever()
# retriever

# retreival_chain = create_retrieval_chain(retriever, document_chain)
# retreival_chain.invoke({"query": "What is CCBI?"})

# from langchain.prompts import PromptTemplate

# prompt_template = """
# Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.

# {context}

# Question: {query}
# Helpful Answer:
# """

# PROMPT = PromptTemplate(
#     template=prompt_template,
#     input_variables=["context", "query"]
# )


# from langchain.chains import RetrievalQA

# # Create the retrieval QA chain with the custom prompt template
# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type="stuff",
#     retriever=vectorstore.as_retriever(),
#     chain_type_kwargs={"prompt": PROMPT},
#     return_source_documents=True,
# )

# # Ask a question
# query = "What research facilities are available in the Computer Science department?"
# result = qa({"query": query})
# print(result['result'])

# from langchain.chains import ConversationalRetrievalChain
# chain = ConversationalRetrievalChain.from_llm(llm, vectorstore.as_retriever(), return_source_documents=True)

# query = "What research facilities are available in the Computer Science department?"
# chat_history = []  # Initialize with empty or previous conversation history

# result = chain({"question": query, "chat_history": chat_history})

# print(result['result'])

# # Optionally print the source documents
# for doc in result['source_documents']:
#     print(doc.page_content)
