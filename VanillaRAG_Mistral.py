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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

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

from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("CS_OpenDay_General.docx")

loaded_documents = loader.load()


def extract_metadata(text):
    sections = text.split('Source:')[1:]  # Split by 'Source:' and remove the first empty part
    all_metadata = []
    
    for section in sections:
        metadata = {}
        lines = section.split('\n')
        metadata['source'] = lines[0].strip()
        
        for line in lines[1:]:
            if line.startswith('Metadata:'):
                metadata['about'] = line.replace('Metadata:', '').strip()
                break
        
        all_metadata.append(metadata)
    
    return all_metadata

# After loading documents
all_metadata = []
for document in loaded_documents:
    # Extract metadata from the document content
    metadata_list = extract_metadata(document.page_content)
    all_metadata.extend(metadata_list)

# text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=30)

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# Add chunk IDs and assign correct metadata to split documents
current_metadata_index = 0
for i, split in enumerate(recreated_splits):
    split.metadata['chunk_id'] = i
    
    # Check if we need to update the current metadata
    if 'Source:' in split.page_content:
        if current_metadata_index < len(all_metadata):
            split.metadata.update(all_metadata[current_metadata_index])
            current_metadata_index += 1
    elif current_metadata_index > 0:
        # For chunks without 'Source:', use the last seen metadata
        split.metadata.update(all_metadata[current_metadata_index - 1])

recreated_splits


vectorstore = Chroma.from_documents(
    recreated_splits,
    embeddings,
    collection_name="CS_OpenDay",
    persist_directory="/home/akash/HSv2/vecdb"
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

# chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
# query = "Are there placements?"
# doc = retriever_vanilla.get_relevant_documents(query)
# results = chain.run(input_documents = doc, question = query)
from langchain.output_parsers import RegexParser
# Define the regex parser to extract the answer
output_parser = RegexParser(
    regex=r"Helpful Answer: \[ANSWER\](.*?)\[\/INST\]",
    output_keys=["answer"]
)

# chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(
    llm=HuggingFacePipeline(pipeline=generate_text),
    chain_type="stuff",
    retriever = retriever_vanilla,
    return_source_documents=True,
    chain_type_kwargs={"prompt": prompt}
)
rag_chain = chain | output_parser

# Use the chain
query = "Are there placements?"
result = rag_chain.invoke({"query": query})

# query = "Are there placements?"
# doc = retriever_vanilla.get_relevant_documents(query)

# rag_chain = (chain | output_parser)

# results = rag_chain.invoke({input_documents:doc}, question=query)
print(result)

from langchain.chains import RetrievalQA

# import time
# from datasets import Dataset
# from tqdm import tqdm
# import pandas as pd

# # Create test set
# testVanilla = pd.read_csv('ROBIN_FINAL_TEST_SET.csv')
# questions = testVanilla['question'].tolist()

# # Create empty lists to store the results and the time taken
# results = []
# retrieval_time_list = []
# chain_time_list = []

# # Loop through each question
# for question in tqdm(questions):
#     # Time the document retrieval process
#     start_retrieval = time.time()
#     doc = retriever_vanilla.get_relevant_documents(question)
#     end_retrieval = time.time()
    
#     retrieval_time = end_retrieval - start_retrieval
#     retrieval_time_list.append(retrieval_time)  # Store retrieval time
    
#     # Time the chain run process
#     start_chain = time.time()
#     result = chain.run(input_documents=doc, question=question)
#     end_chain = time.time()
    
#     chain_time = end_chain - start_chain
#     chain_time_list.append(chain_time)  # Store chain run time
    
#     results.append(result)

# # Create a pandas DataFrame to store the results and times taken
# df = pd.DataFrame({
#     "Question": questions,
#     "Answer": results,
#     "Retrieval_Time": retrieval_time_list,
#     "Chain_Time": chain_time_list
# })
# df.to_csv('Results_Vanilla.csv', index=False)
