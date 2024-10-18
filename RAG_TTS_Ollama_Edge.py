import ollama
from chromadb import Client
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain_community.document_loaders import Docx2txtLoader
from typing import List
import torch

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

llm = ollama.chat(model="mistral")  

embeddings = ollama.embeddings(
    model="nomic-embed-text",
)  

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
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1900, chunk_overlap=128) 
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
vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings, collection_name="ROBIN-2")

retriever_vanilla = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 2})
retriever_mmr = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 2})
retriever_BM25 = BM25Retriever.from_documents(recreated_splits, search_kwargs={"k": 2})

# initialize the ensemble retriever with 3 Retrievers
ensemble_retriever = EnsembleRetriever(
    retrievers=[retriever_vanilla, retriever_mmr, retriever_BM25], weights=[0.4, 0.4, 0.2]
)


class QuestionMemory:
    """Keeps track of the last few questions asked for context during paraphrasing."""
    def __init__(self, max_questions: int = 5):
        self.questions: List[str] = []
        self.max_questions = max_questions

    def add_question(self, question: str):
        """Adds a question to history, maintaining a maximum of `max_questions`."""
        self.questions.append(question)
        if len(self.questions) > self.max_questions:
            self.questions.pop(0)

    def get_history(self) -> str:
        """Returns the question history as a string."""
        return "\n".join(self.questions)

# Initialize question memory
question_memory = QuestionMemory()

class OllamaRAG:
    def __init__(self, retriever,llm):
        self.retriever = retriever 
        self.question_memory = question_memory
        self.llm = llm

    def paraphrase_question(self, new_question):
        """Paraphrase the user's question using the provided paraphrase prompt template."""
        question_history = self.question_memory.get_history()
        paraphrase_prompt = f"""
        [INST]
        You are an advanced AI assistant for Nottingham Trent University's Computer Science Department, specializing in generating optimal questions for a Retrieval-Augmented Generation (RAG) system.This RAG system is called ROBIN. Your task is to analyze the question history and the new question, then produce a refined version that maximizes relevance for semantic search, keyword search, and BM25 ranking, while aligning with the specific data structure used.

        Question History:
        {question_history}

        New Question: {new_question}

        Refined Question for RAG:
        [/INST]
        """

        response = ollama.generate(
            model=self.llm,
            prompt=paraphrase_prompt,
            stream=False
        )
        
        return response['text'].strip()

    def generate_answer(self, context, question):
        """Generate the answer using a specific RAG prompt template."""
        answer_prompt = f"""
        [INST]
        You are "AI Robin Hood," an assistant at Nottingham Trent University's (NTU) Open Day at Clifton Campus, Nottingham, UK.

        STRICT RESPONSE PROTOCOL:
        1. First, carefully check if the provided context contains information relevant to the question.
        2. If the context DOES NOT contain the required information:
           - DO NOT make assumptions or create information
           - DO NOT use general knowledge about universities
           - Respond ONLY with: "Me scholar, I do not have that information at the moment."

        CONTEXT: {context}
        QUESTION: {question}
        AI Robin Hood's Answer: [/INST]
        """

        response = ollama.generate(
            model="mistral",
            prompt=answer_prompt,
            stream=False
        )
        
        return response['text'].strip()

    def run_rag(self, question):
        """Main function to paraphrase the question, retrieve documents, and generate an answer."""
        # Step 1: Paraphrase the question using history
        paraphrased_question = self.paraphrase_question(question)
        print(f"Paraphrased Question: {paraphrased_question}")

        # Step 2: Retrieve documents using the paraphrased question
        retrieved_docs = self.retriever.invoke(paraphrased_question)
        print(f"Retrieved Documents: {retrieved_docs}")

        # Step 3: Generate answer using the retrieved context and paraphrased question
        final_answer = self.generate_answer(retrieved_docs, paraphrased_question)

        # Step 4: Add the current question to the history
        self.question_memory.add_question(question)

        return final_answer

# Example usage:
# Assuming retriever is an instance of a vectorstore retriever with a .invoke() method
rag_system = OllamaRAG(ensemble_retriever,llm)

user_question = "What are the best courses at NTU?"
answer = rag_system.run_rag(user_question)
print(f"Final Answer: {answer}")

