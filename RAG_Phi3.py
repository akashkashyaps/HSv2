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

# Test the model with a prompt
input_text = "what features does nottingham trent university computer science department have?"
generated_ids = generate_text(input_text)[0]['generated_text']
print(generated_ids)


from langchain.document_loaders import SeleniumURLLoader

urls = ["https://www.ntu.ac.uk/course/computer-science", 
        "https://www.ntu.ac.uk/study-and-courses/academic-schools/science-and-technology/Computer-Science"]
       
loader = SeleniumURLLoader(urls)
documents = loader.load()

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
all_splits = text_splitter.split_documents(documents)

# Define the model name and kwargs for embeddings
model_name = "sentence-transformers/all-mpnet-base-v2"
model_kwargs = {"device": "cuda"}

# Create the HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

# Create the vectorstore
vectorstore = Chroma.from_documents(
    all_splits,
    embeddings,
    persist_directory="/home/akash/HSv2"
)

retriever = vectorstore.as_retriever()


prompt_template =("""
<|user|>
Use the following pieces of context to answer the question at the end. If you don't know the answer, just say that you don't know, don't try to make up an answer.
{context}
NTU is Nottingham Trent University.It is not Singapore. Note this it is very important.
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



# Load test set
testset3 = pd.read_csv('testset3.csv')

# Initialize lists to store results
retrieved_contexts = []
answers = []
error_indices = []

# Iterate through each question in the DataFrame
for index, row in testset3.iterrows():
    try:
        question = row['question']
        
        # Debugging print to ensure correct row access
        print(f"Processing question {index + 1}/{len(testset3)}: {question}")
        
        # Retrieve relevant documents/context for the question
        docs = vectorstore.similarity_search(question)  # Perform similarity search
        
        # Debugging print to inspect retrieved documents
        print(f"Retrieved {len(docs)} documents for question {index + 1}")
        
        # Concatenate the retrieved documents into a single context string
        retrieved_context = " ".join([doc.page_content for doc in docs])
        retrieved_contexts.append(retrieved_context)
        
        # Run the RAG chain with the retrieved context and the question
        result = chain.run(input_documents=docs, question=question)
        
        # Split the result into context and answer
        if '\nHelpful Answer:' in result:
            context, answer = result.split('\nHelpful Answer:', 1)
            context = context.strip()
            answer = answer.strip()
        else:
            context = retrieved_context
            answer = "I don't know"
        
        # Debugging print to inspect the result
        print(f"Generated context for question {index + 1}: {context}")
        print(f"Generated answer for question {index + 1}: {answer}")
        
        answers.append(answer)
        
    except Exception as e:
        print(f"An error occurred with question {index + 1}: {e}")
        retrieved_contexts.append(None)  # Append None to keep list length consistent
        answers.append(None)  # Append None to keep list length consistent
        error_indices.append(index)

# Ensure the lengths of the lists match the DataFrame
while len(retrieved_contexts) < len(testset3):
    retrieved_contexts.append(None)
while len(answers) < len(testset3):
    answers.append(None)

# Add the results to the DataFrame
testset3['RAG Context'] = retrieved_contexts
testset3['RAG Answer'] = answers

# Save the DataFrame with the results
testset3.to_csv('testset3_with_results.csv', index=False)