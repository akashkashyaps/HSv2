from ragas.testset.generator import TestsetGenerator  
from ragas.testset.evolutions import simple, reasoning, multi_context, conditional  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
# from langchain.document_loaders import SeleniumURLLoader  
from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings
import json

# # Define the model name and kwargs for embeddings
# model_name = "sentence-transformers/all-mpnet-base-v2"
# model_kwargs = {"device": "cuda"}

# from langchain.embeddings import HuggingFaceEmbeddings

# # Create the HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# from langchain.vectorstores import Chroma
# vectordb = Chroma(persist_directory='/home/akash/HSv2/', embedding_function=embeddings)
# retriever = vectordb.as_retriever()
# query = "Are there any connections with local employers?"
# doc = vectordb.similarity_search(query)


# urls = ["https://www.ntu.ac.uk/course/computer-science", 
#     "https://www.ntu.ac.uk/study-and-courses/academic-schools/science-and-technology/Computer-Science"]

# loader = SeleniumURLLoader(urls)  
# documents = loader.load() 

# for document in documents:
#     document.metadata["filename"] = document.metadata["source"]  # Adding a "filename" key to the metadata dictionary of each document

# # Save the documents locally as JSON
# def save_documents_to_json(documents, filename):
#     serializable_data = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in documents]
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(serializable_data, f, ensure_ascii=False, indent=4)

# save_documents_to_json(documents, "documents.json")

from langchain.schema import Document
# Load JSON data from file
# # def load_documents_from_json(filename):
# #     with open(filename, 'r', encoding='utf-8') as f:
# #         data = json.load(f)
# #     return [Document(page_content=doc["page_content"], metadata=doc["metadata"]) for doc in data]

# # # Load the documents
# # loaded_documents = load_documents_from_json("documents.json")
# import re
# from docx import Document as DocxDocument
# from langchain.schema import Document

# def load_documents_from_docx(filename):
#     docx = DocxDocument(filename)
#     full_text = []
#     metadata = {}
#     documents = []

#     current_metadata = None
#     current_text = []

#     # Define regex patterns
#     source_pattern = re.compile(r"^Source:\s*(.*)$")
#     metadata_pattern = re.compile(r"^Metadata:\s*(.*)$")
#     text_pattern = re.compile(r"^Text:\s*$")

#     for para in docx.paragraphs:
#         text = para.text.strip()

#         if source_pattern.match(text):
#             # Save the previous document if it exists
#             if current_metadata and current_text:
#                 documents.append(Document(page_content="\n".join(current_text), metadata=current_metadata))
#                 current_text = []

#             current_metadata = {"source": source_pattern.match(text).group(1)}
        
#         elif metadata_pattern.match(text):
#             if current_metadata is not None:
#                 current_metadata["metadata"] = metadata_pattern.match(text).group(1)
        
#         elif text_pattern.match(text):
#             continue  # Skip the "Text:" line
        
#         else:
#             if current_metadata is not None:
#                 current_text.append(text)
    
#     # Add the last document
#     if current_metadata and current_text:
#         documents.append(Document(page_content="\n".join(current_text), metadata=current_metadata))
    
#     return documents

# # # Verify the first loaded document
# # if loaded_documents:
# #     print("First loaded document:")
# #     print(f"Content: {loaded_documents[0].page_content}")
# #     print(f"Metadata: {loaded_documents[0].metadata}")
# # else:
# #     print("No documents found in the loaded data.")

# loaded_documents = load_documents_from_docx("CS_OpenDay_General.docx")
# for doc in loaded_documents:
#     print(f"Page Content: {doc.page_content}")
#     print(f"Metadata: {doc.metadata}")
#     print("\n----------------------\n")

from langchain_community.document_loaders import Docx2txtLoader

loader = Docx2txtLoader("CS_OpenDay_General.docx")

loaded_documents = loader.load()


for document in loaded_documents:
    document.metadata['filename'] = document.metadata['source']

# Recreate the text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# # Split the documents into chunks
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)  
# all_splits = text_splitter.split_documents(documents) 

# # Save the chunks locally as JSON
# def save_to_json(data, filename):
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(data, f, ensure_ascii=False, indent=4)

# # Convert documents to a serializable format
# serializable_data = [{"text": doc.page_content, "metadata": doc.metadata} for doc in all_splits]

# # Save to JSON file
# save_to_json(serializable_data, "document_chunks.json")

import torch

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

generator_llm = Ollama(model="phi3")  # Creating an instance of Ollama with the "phi3" model
critic_llm = Ollama(model="llama3")  # Creating an instance of Ollama with the "llama3" model

ollama_emb = OllamaEmbeddings(
    model="nomic-embed-text",
)  # Creating an instance of OllamaEmbeddings with the "nomic-embed-text" model

print(generator_llm.invoke('Say hello'))  
print(critic_llm.invoke('Say hello'))  

# r2 = ollama_emb.embed_query(
#     "What is the second letter of Greek alphabet"
# )  

# print(f'Embedding dimension: {len(r2)}')  

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=ollama_emb,
) 
# documents = vectordb.get()

# generate testset
testset = generator.generate_with_langchain_docs(recreated_splits, test_size=10, distributions={simple: 0.5, reasoning: 0.20, multi_context: 0.15, conditional: 0.15 }, raise_exceptions=False)  # Generating a testset using the generator and the chunks of documents

test_df = testset.to_pandas()  
test_df.to_csv('ROBIN_testset1.csv', index=False)  
import pandas as pd

# # Function to generate and save test sets in chunks
# def generate_and_save_testsets(splits, total_samples, filename_prefix):
#     chunk_size = len(splits) // (total_samples // 100)  # Calculate chunk size to distribute total samples evenly
#     remaining_samples = total_samples
#     test_dfs = []

#     for i in range(0, len(splits), chunk_size):
#         chunk = splits[i:i + chunk_size]
#         if remaining_samples <= 0:
#             break
#         current_chunk_size = min(len(chunk), remaining_samples)
#         testset = generator.generate_with_langchain_docs(chunk, test_size=current_chunk_size, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, raise_exceptions=False)
#         test_df = testset.to_pandas()
#         test_dfs.append(test_df)
#         test_df.to_csv(f'{filename_prefix}_chunk_{i//chunk_size + 1}.csv', index=False)
#         remaining_samples -= current_chunk_size

#     # Combine all chunks into a single DataFrame
#     combined_df = pd.concat(test_dfs, ignore_index=True)
#     combined_df.to_csv(f'{filename_prefix}_combined.csv', index=False)

# # Generate and save test sets ensuring 1000 samples
# generate_and_save_testsets(recreated_splits, total_samples=1000, filename_prefix='testset5')


# # Function to generate and save test sets in chunks
# def generate_and_save_testsets(splits, total_samples, filename_prefix, chunk_size=100):
#     remaining_samples = total_samples
#     test_dfs = []

#     for i in range(0, len(splits), chunk_size):
#         chunk = splits[i:i + chunk_size]
#         if remaining_samples <= 0:
#             break
#         current_chunk_size = min(len(chunk), remaining_samples)
#         testset = generator.generate_with_langchain_docs(chunk, test_size=current_chunk_size, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, raise_exceptions=False)
#         test_df = testset.to_pandas()
#         test_dfs.append(test_df)
#         test_df.to_csv(f'{filename_prefix}_chunk_{i//chunk_size + 1}.csv', index=False)
#         remaining_samples -= current_chunk_size

#     # Combine all chunks into a single DataFrame
#     combined_df = pd.concat(test_dfs, ignore_index=True)
#     combined_df.to_csv(f'{filename_prefix}_combined.csv', index=False)

# # Generate and save test sets ensuring 5 samples
# generate_and_save_testsets(recreated_splits, total_samples=1000, filename_prefix='testset6', chunk_size=100)