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
from langchain.prompts import PromptTemplate
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=2048, chunk_overlap=128) 
# TODO: Check the chunk size and overlap

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

import os 

home_directory = os.path.expanduser("~")
persist_directory = os.path.join(home_directory, "HSv2", "vecdb")
vectorstore = Chroma.from_documents(
    recreated_splits,
    embeddings,
    collection_name="ROBIN-2",
    persist_directory=persist_directory
)

retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
retriever_BM25 = BM25Retriever.from_documents(recreated_splits, search_kwargs={"k": 2})

# initialize the ensemble retriever with 3 Retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.4, 0.4, 0.2]
)


from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from typing import List
import re

class QuestionMemory:
    def __init__(self, max_questions: int = 5):
        self.questions: List[str] = []
        self.max_questions = max_questions

    def add_question(self, question: str):
        self.questions.append(question)
        if len(self.questions) > self.max_questions:
            self.questions.pop(0)

    def get_history(self) -> str:
        return "\n".join(self.questions)

question_memory = QuestionMemory()

paraphrase_template = ("""
[INST]
You are an advanced AI assistant for Nottingham Trent University's Computer Science Department, specializing in generating optimal questions for a Retrieval-Augmented Generation (RAG) system.This RAG system is called ROBIN. Your task is to analyze the question history and the new question, then produce a refined version that maximizes relevance for semantic search, keyword search, and BM25 ranking, while aligning with the specific data structure used.

Guidelines:
1. Assess if the new question is related to the question history.
2. For related questions:
   a. Incorporate crucial context from the history.
   b. Maintain the core intent of the new question.
3. For unrelated questions:
   a. Focus on enhancing the question for search relevance without adding historical context.
4. In all cases:
   a. Use specific, descriptive terms that align with potential content and metadata in the database.
   b. Include full entity names and relevant abbreviations (e.g., "Nottingham Trent University (NTU)").
   c. Structure the question to support both semantic understanding and keyword matching.
   d. Ensure the question is self-contained and understandable without additional context.
   e. When applicable, include terms that might appear in the 'Source:' or 'Metadata:' fields of the documents.
   f. Frame questions to target information that could be contained within 300-character chunks.
   g. Make sure the question has some synonyms of the keywords in addition to the keywords themselves to improve search results.
5. Students are usually present students or prospective students or previous students (graduates) from Nottingham Trent University.
6. If the question is not related to the university or the Computer Science department, do not change the question, return as it is.
7. Do not introduce speculative information or assumptions.
8. Generate only one refined question per input.

Examples to learn from:
New Question: "Who is the HOD?"
Refined Question for RAG: "Who is the head of the Computer Science department at Nottingham Trent University?"

New Question: "where can i get food from?"
Refined Question for RAG: "Where can students find food on the Clifton Campus of Nottingham Trent University?"
                       
New Question: "where is the nicest place to travel in the winter when you want to get some sun?"
Refined Question for RAG: "Where is the nicest place to travel in the winter when you want to get some sun?"

New Question: "How do I bake a cake? Give me a recipe."
Refined Question for RAG: "How do I bake a cake? Give me a recipe."
                       
New Question: "Can you hear me?"
Refined Question for RAG: "Can you hear me?"
                       
Question History:
{question_history}

New Question: {question}

Refined Question for RAG:
[/INST]
""")

paraphrase_prompt = PromptTemplate(template=paraphrase_template, input_variables=["question_history", "question"])

rag_template = ("""
[INST]
You are "AI Robin Hood," an assistant at Nottingham Trent University's (NTU) Open Day at Clifton Campus, Nottingham, UK.

STRICT RESPONSE PROTOCOL:
1. First, carefully check if the provided context contains information relevant to the question.
2. If the context DOES NOT contain the required information:
   - DO NOT make assumptions or create information
   - DO NOT use general knowledge about universities
   - Respond ONLY with: "By my honor as keeper of Sherwood Forest, I find no such information in my scrolls about [topic]."

3. If the context DOES contain relevant information:
   - Use a mix of modern and slightly archaic English (using "ye," "thy," "Aye," "Nay")
   - Keep responses brief (3-4 sentences maximum)
   - Refer to students as "merry scholars"
   - Base EVERY detail strictly on the provided context

Character Elements:
- Mix modern and medieval English while maintaining clarity
- Use "ye" instead of "you"
- Use "thy" for "your"
- Refer to students as "merry scholars"
- NO pirate speech ("me hearty" or "ye olde")

Remember: Like a true archer, you must only hit targets you can see (information in the context). If you cannot see it, you must not shoot (respond).Never fabricate or assume information not present in the context.

CONTEXT: {context}
QUESTION: {question}
AI Robin Hood's Answer: [/INST]
""")

prompt = PromptTemplate(template=rag_template, input_variables=["context", "question"])

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

paraphrase_chain = paraphrase_prompt| llm |StrOutputParser()

rag_chain = prompt | llm | StrOutputParser()

from langchain_core.runnables import RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from llm_guard.input_scanners import PromptInjection, BanTopics, Toxicity as InputToxicity
from llm_guard.input_scanners.prompt_injection import MatchType as InputMatchType
from llm_guard.output_scanners import Toxicity as OutputToxicity, NoRefusal, BanTopics
from llm_guard.output_scanners.toxicity import MatchType as OutputMatchType

# Initialize the Prompt Injection scanner
prompt_injection_scanner = PromptInjection(threshold=0.92, match_type=InputMatchType.FULL)

# Initialize the Toxicity scanner for inputs
input_toxicity_scanner = InputToxicity(threshold=0.9, match_type=InputMatchType.SENTENCE)

# Initialize the Toxicity scanner for outputs
output_toxicity_scanner = OutputToxicity(threshold=0.9, match_type=OutputMatchType.SENTENCE)

# Initialize the No Refusal scanner
no_refusal_scanner = NoRefusal(threshold=0.9, match_type=OutputMatchType.FULL)

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

    # Step 3: Get the question history from the memory
    question_history = question_memory.get_history()

    with torch.no_grad():
        # Step 4: Paraphrase the sanitized query using question history
        paraphrased_output = paraphrase_chain.invoke({"question": sanitized_query, "question_history": question_history}, config={"callbacks": [langfuse_handler]})
        paraphrased_query = extract_answer_instance.run(paraphrased_output)

        # Step 5: If paraphrasing fails, use the original sanitized query
        if not paraphrased_query:
            paraphrased_query = sanitized_query

        # Step 6: Store the original (or paraphrased) query in the memory for future use
        question_memory.add_question(sanitized_query)

        # Step 7: Retrieve context from vector store using the paraphrased (or original) query
        context = ensemble_retriever.get_relevant_documents(sanitized_query)

        # Step 8: Generate a response using the RAG pipeline with the paraphrased (or original) query
        result = rag_chain.invoke({"question": paraphrased_query, "context": context}, config={"callbacks": [langfuse_handler]})

    # Step 9: Debug print to check the structure of the result
    print("Debug - Result structure:", result)

    # Step 10: Extract the answer from the result
    answer = extract_answer_instance.run(result)

    # Step 11: Sanitize the output before returning
    sanitized_answer = scan_output(paraphrased_query, answer)
    
    return sanitized_answer

if __name__ == "__main__":
    print(get_rag_response("What is the history of Nottingham Trent University?"))

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
        
#     # Time the chain run process
#     start_chain = time.time()
#     result = chain.invoke(question)
#     end_chain = time.time()
    
#     chain_time = end_chain - start_chain
#     chain_time_list.append(chain_time)  # Store chain run time
    
#     results.append(result)

# # Create a pandas DataFrame to store the results and times taken
# df = pd.DataFrame({
#     "Question": questions,
#     "Answer": results,
#     "Chain_Time": chain_time_list
# })
# df.to_csv('Fusion_Mistral7B.csv', index=False)