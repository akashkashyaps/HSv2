# Import dependencies
import os
import torch
import transformers
import pandas as pd
import numpy as np
from torch import cuda, bfloat16
from langchain_huggingface import HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.prompts import PromptTemplate
from datasets import Dataset
from langchain_community.document_loaders import Docx2txtLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

LANGFUSE_SECRET_KEY = "sk-lf-..."
LANGFUSE_PUBLIC_KEY = "pk-lf-..."
LANGFUSE_HOST = "https://cloud.langfuse.com"

from langfuse.callback import CallbackHandler
langfuse_handler = CallbackHandler(
    public_key="pk-lf-7891f375-f1da-47ff-94a9-0a715b95012c",
    secret_key="sk-lf-033efc71-3409-4e9f-9670-713e9a6889a1",
    host="https://cloud.langfuse.com"
)

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
    tokenizer.convert_tokens_to_ids("")
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

loader = Docx2txtLoader("CS_OpenDay_General_Updated.docx")

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


home_directory = os.path.expanduser("~")
persist_directory = os.path.join(home_directory, "HSv2", "vecdb")

vectorstore = Chroma.from_documents(
    recreated_splits,
    embeddings,
    collection_name="CS_OpenDay",
    persist_directory=persist_directory
)

retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = ("""
[INST]

For the purposes of this conversation, you are a helpful agent present at a University Open Day at Nottingham Trent University (NTU) at Clifton Campus, Nottingham, United Kingdom. You will take on the persona of “AI Robin Hood”, channeling the language and style of the legendary Robin Hood which is very important. A University Open Day is an event where future University students visit to see the campus, facilities, and meet the teaching staff. The future students are going to ask you questions about the University which you will answer in the tone and style of Robin Hood, as he might have spoken. If you do not know the answer, just say that you cannot answer the question, and do not try to make up an answer. Please keep your answers brief, as this is a conversation taking place in real-time.
Recent conversations:
{chat_history}

CONTEXT: {context}
QUESTION: {question}
Helpful Answer: [/INST]
""")

prompt = PromptTemplate(template=prompt_template, input_variables=["memory", "context", "question"])


llm = HuggingFacePipeline(pipeline=generate_text)

import re

class ExtractAnswer:
    def run(self, text):
        # Adjust the regex pattern to handle the potential characters and spacing around [/INST]
        match = re.search(r'\[\/INST\]\s*(.*)', text, re.DOTALL)
        if match:
            answer = match.group(1).strip().replace("\n", " ").replace("\r", "").replace("[/", "").replace("]", "")
            return answer
        else:
            return None

# Define an instance of ExtractAnswer
extract_answer_instance = ExtractAnswer()

class ChatHistoryManager:
    def __init__(self, history_limit=10):
        self.history_limit = history_limit
        self.chat_history = []

    def add_interaction(self, question, answer):
        if len(self.chat_history) >= self.history_limit * 2:  # Each Q&A is 2 entries
            self.chat_history = self.chat_history[2:]  # Remove the oldest Q&A pair
        self.chat_history.append(f"Q: {question}")
        self.chat_history.append(f"A: {answer}")

    def get_chat_history(self):
        return "\n".join(self.chat_history)
    
    def clear_history(self):
        self.chat_history = []

# Initialize the chat history manager
history_manager = ChatHistoryManager()

from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
rag_chain = prompt | llm | StrOutputParser()
from llm_guard.input_scanners import PromptInjection, BanTopics, Toxicity as InputToxicity
from llm_guard.input_scanners.prompt_injection import MatchType as InputMatchType
from llm_guard.output_scanners import Toxicity as OutputToxicity, NoRefusal, BanTopics
from llm_guard.output_scanners.toxicity import MatchType as OutputMatchType

# Initialize the Prompt Injection scanner
prompt_injection_scanner = PromptInjection(threshold=0.92, match_type=InputMatchType.FULL)

# Initialize the Toxicity scanner for inputs
input_toxicity_scanner = InputToxicity(threshold=0.92, match_type=InputMatchType.SENTENCE)

# Initialize the Toxicity scanner for outputs
output_toxicity_scanner = OutputToxicity(threshold=0.92, match_type=OutputMatchType.SENTENCE)

# Initialize the No Refusal scanner
no_refusal_scanner = NoRefusal(threshold=0.92, match_type=OutputMatchType.FULL)

# Initialize the Ban Topics scanner
ban_topics_scanner = BanTopics(topics=["violence", "politics", "religion"], threshold=0.75)

def scan_input(prompt):
    # Scan for prompt injection
    sanitized_prompt, is_valid, _ = prompt_injection_scanner.scan(prompt)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    # Scan for banned topics
    sanitized_prompt, is_valid, _ = ban_topics_scanner.scan(prompt, sanitized_prompt)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    # Scan for toxicity
    sanitized_prompt, is_valid, _ = input_toxicity_scanner.scan(sanitized_prompt)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    return sanitized_prompt

def scan_output(prompt, model_output):
    # Scan for output toxicity
    sanitized_output, is_valid, _ = output_toxicity_scanner.scan(prompt, model_output)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    # Scan for no refusal
    sanitized_output, is_valid, _ = no_refusal_scanner.scan(prompt, sanitized_output)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    # Scan for banned topics
    sanitized_output, is_valid, _ = ban_topics_scanner.scan(prompt, sanitized_output)
    if not is_valid:
        return "Sorry, I'm just an AI hologram, can I help you with something else."

    return sanitized_output

def get_rag_response(query):
    # Step 1: Sanitize the input query
    sanitized_query = scan_input(query)
    
    # Step 2: Check if the sanitized query is valid
    if sanitized_query == "Sorry, I'm just an AI hologram, can I help you with something else.":
        return sanitized_query

    # Step 3: Retrieve context from vector store using sanitized query
    context = retriever_vanilla.get_relevant_documents(sanitized_query)
    
    # Step 4: Get chat history
    chat_history = history_manager.get_chat_history()

    # Step 5: Generate a response using the RAG pipeline
    result = rag_chain.invoke({"context": context, "chat_history": chat_history, "question": sanitized_query},config={"callbacks": [langfuse_handler]})

    # Step 6: Extract the answer from the result
    answer = extract_answer_instance.run(result)

    # Step 7: Sanitize the output before returning
    # TODO: restore this step when common outputs stop being flagged as violent
    sanitized_answer = scan_output(sanitized_query, answer)
    
    # Step 8: Store the sanitized Q&A pair in chat history
    history_manager.add_interaction(sanitized_query, sanitized_answer)
    
    return answer

if __name__ == "__main__":
    print(get_rag_response("What is the history of Nottingham Trent University?"))