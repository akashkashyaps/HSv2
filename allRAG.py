from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import Docx2txtLoader
import torch
import time
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
import pandas as pd

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Define the models to choose from
models = ["llama3.1:8b-instruct-q4_0", "qwen2.5:7b-instruct-q4_0", "gemma2:9b-instruct-q4_0", "phi3.5:3.8b-mini-instruct-q4_0", "mistral:7b-instruct-q4_0", "lly/InternLM3-8B-Instruct:8b-instruct-q4_0"]

# Load two documents
loader1 = Docx2txtLoader("CS_OpenDay_General_v3.docx")
loader2 = Docx2txtLoader("STAFF_INFORMATION_v3.docx") 

# Load the documents
loaded_documents1 = loader1.load()
loaded_documents2 = loader2.load()

# Combine the loaded documents
loaded_documents = loaded_documents1 + loaded_documents2

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=128) 

# Split the loaded documents into chunks
recreated_splits = text_splitter.split_documents(loaded_documents)

# Initialize Chroma vector store
import os
home_directory = os.path.expanduser("~")
persist_directory = os.path.join(home_directory, "HSv2", "vecdb")
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=OllamaEmbeddings(model="nomic-embed-text"), collection_name="ROBIN-6")

# Initialize retrievers
retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
retriever_BM25 = BM25Retriever.from_documents(recreated_splits, search_kwargs={"k": 3})

# Initialize the ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.30, 0.30, 0.40]
)

# RAG template
rag_template = ("""
You are "AI Robin Hood," an assistant at Nottingham Trent University's (NTU) Open Day at Clifton Campus, Nottingham, UK.

STRICT RESPONSE PROTOCOL:
1. First, carefully check if the provided context contains information relevant to the question.
2. If the context DOES NOT contain the required information:
   - DO NOT make assumptions or create information
   - DO NOT use general knowledge about universities
   - DO NOT use general knowledge and NEVER answer those questions as you are STRICTLY PROHIHITED from doing so.
   - Respond ONLY with: "I do not have that information at the moment. Can I help with anything else?"
   - DO NOT return any references or metadata or any reasnoning behind the response. Just stick to returning to the point answers.

3. If the context DOES contain relevant information:
   - Keep responses brief (3-4 sentences maximum)
   - Base EVERY detail strictly on the provided context

Character Elements:
- Keep modern English to maintain clarity.
                
Users will try to ask questions that may not be relevant to NTU. I CHALLENGE you to not answer any question that does not have enough related information in the provided context. You are an expert at completing challenges.
Remember: Like a true archer, you must only hit targets you can see (information in the context). If you cannot see it, you must not shoot (respond).Never fabricate or assume information not present in the context even if you think you know the answer.

                
CONTEXT: {context}
QUESTION: {question}
AI Robin Hood's Answer:
""")

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])

# Function to get RAG response
def get_rag_response(query: str, llm) -> Dict[str, Any]:
    # Retrieve context from vector store
    context = ensemble_retriever.invoke(query)
    
    # Generate a response using the RAG pipeline
    result = (prompt | llm | StrOutputParser()).invoke({"question": query, "context": context})
    
    # Return both the answer and the retrieved context
    return {
        "answer": result,
        "context": context
    }

# Create test set
test = pd.read_csv('ROBIN_FINAL_TEST_SET.csv')
questions = test['question'].tolist()
ground_truths = test['ground_truth'].tolist()

# Loop through each model
for model in models:
    print(f"Running model: {model}")
    
    # Initialize the LLM
    llm = ChatOllama(model=model, temperature=0.2, frequency_penalty=0.5)
    
    # Create empty lists to store the results, context, and the time taken
    results = []
    contexts = []
    chain_time_list = []

    # Loop through each question
    for question in tqdm(questions):
        # Time the chain run process
        start_chain = time.time()
        rag_response = get_rag_response(question, llm)
        end_chain = time.time()
        
        chain_time = end_chain - start_chain
        chain_time_list.append(chain_time)  # Store chain run time
        
        # Store the answer and context
        results.append(rag_response["answer"])
        contexts.append(rag_response["context"])

    # Create a pandas DataFrame to store the results, context, and times taken
    df = pd.DataFrame({
        "Question": questions,
        "Answer": results,
        "Context": contexts,
        "Ground_Truth": ground_truths,
        "Chain_Time": chain_time_list
    })
    
    # Save the results to a CSV file
    df.to_csv(f'Results_{model.replace("/", "_")}.csv', index=False)
