from ragas.testset.generator import TestsetGenerator  
from ragas.testset.evolutions import simple, reasoning, multi_context  
from langchain.text_splitter import RecursiveCharacterTextSplitter  
from langchain.document_loaders import SeleniumURLLoader  
from langchain_community.llms import Ollama  
from langchain_community.embeddings import OllamaEmbeddings  

urls = ["https://www.ntu.ac.uk/course/computer-science", 
    "https://www.ntu.ac.uk/study-and-courses/academic-schools/science-and-technology/Computer-Science"]

loader = SeleniumURLLoader(urls)  
documents = loader.load() 

for document in documents:
    document.metadata["filename"] = document.metadata["source"]  # Adding a "filename" key to the metadata dictionary of each document

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)  
all_splits = text_splitter.split_documents(documents) 

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

r2 = ollama_emb.embed_query(
    "What is the second letter of Greek alphabet"
)  

print(f'Embedding dimension: {len(r2)}')  

generator = TestsetGenerator.from_langchain(
    generator_llm=generator_llm,
    critic_llm=critic_llm,
    embeddings=ollama_emb,
) 

# generate testset
testset = generator.generate_with_langchain_docs(all_splits, test_size=300, distributions={simple: 0.5, reasoning: 0.25, multi_context: 0.25}, raise_exceptions=False)  # Generating a testset using the generator and the chunks of documents

test_df = testset.to_pandas()  
test_df.to_csv('testset4.csv', index=False)  
