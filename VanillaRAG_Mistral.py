# Import dependancies
import torch
import transformers
import pandas as pd
import numpy as np
from torch import cuda, bfloat16
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datasets import Dataset
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker


# Define the model
model_id = "mistralai/Mistral-7B-Instruct-v0.3"
hf_auth = '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'
tokenizer = transformers.AutoTokenizer.from_pretrained(
    model_id,
    token=hf_auth,
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
    token=hf_auth,
    trust_remote_code=True
)
model = transformers.AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    config=model_config,
    quantization_config=bnb_config,
    device_map='auto',
    token=hf_auth,
)   

terminators = [
    tokenizer.eos_token_id,
    tokenizer.convert_tokens_to_ids("<|eot_id|>")
]

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
    repetition_penalty=1.1,
    eos_token_id=terminators,
)

# Define the model name and kwargs for embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# Create the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

from langchain_community.document_loaders import UnstructuredWordDocumentLoader

docx_file_path = "CS_OpenDay_General.docx"
loader = UnstructuredWordDocumentLoader(docx_file_path, mode="elements")
data = loader.load()
data

# loader = Docx2txtLoader("CS_OpenDay_General.docx")
# documents = loader.load()

text_splitter = SemanticChunker(
    embeddings, breakpoint_threshold_type="percentile"
)
docs = text_splitter.create_documents(data)

vectorstore = Chroma.from_documents(
    docs,
    embeddings,
    collection_name="CS_OpenDay_General",
    persist_directory="/home/akash/HSv2"
)

retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = ("""
[INST]

For the purposes of this conversation, you are a helpful agent who is present at a University Open Day at Nottingham Trent University (NTU) at Clifton Campus, Nottingham, United Kingdom. A University Open Day is an event where future University students will visit to see the campus, facilities, and meet the teaching staff. The future students are going to ask you questions about University which you will answer by using the helpful context at the end of this message. Note that it is very important that you are at Nottingham Trent University (NTU) in the United Kingdom and NOT Nanyang Technological University in Singapore. You will now be given context and asked a question. Your task is to answer the question. If you do not know the answer, just say that you cannot answer the question, do not try to make up an answer.
<|eot_id|>

CONTEXT: {context}
QUESTION: {question}
Helpful Answer: [/INST]
""")


prompt=PromptTemplate(template=prompt_template,input_variables=["context","question"])

llm = HuggingFacePipeline(pipeline=generate_text)

chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
query = "Are there any connections with employers?"
doc = retriever_vanilla.get_relevant_documents(query)
chain.run(input_documents = doc, question = query)

