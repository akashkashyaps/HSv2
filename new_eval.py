from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from langchain_ollama import OllamaEmbeddings
import time
from typing import List, Dict, Any
from datasets import Dataset
from tqdm import tqdm
import pandas as pd
import ast
from langchain_core.output_parsers import StrOutputParser
import re
from pathlib import Path 


context_precision_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Context Precision metric based on the inputs provided below.

Inputs:
User Input: {user_input}
Reference: {reference}
Retrieved Contexts: {retrieved_contexts}

Definition of Context Precision:
Context Precision measures the proportion of relevant chunks within the retrieved contexts. It is computed as the average of the precision@k values for each retrieved context chunk.

For a list of retrieved context chunks:
- Let N be the total number of chunks in retrieved_contexts.
- For each rank k (where 1 ≤ k ≤ N), define:
    precision@k = (Number of relevant chunks among the top k) / k.
A chunk is considered relevant if it provides information that aligns with the Reference (and supports answering the User Input).
- Let rₖ be the relevance indicator for the chunk at rank k, where rₖ = 1 if the chunk is relevant and 0 if it is not.
Then, Context Precision = (1/N) * (precision@1 + precision@2 + ... + precision@N).

Your Task:
1. For each chunk in retrieved_contexts, determine whether it is relevant to the Reference (and consider the User Input if needed). Assign rₖ = 1 for relevant chunks and rₖ = 0 for non-relevant ones.
2. For each rank k from 1 to N, compute:
    precision@k = (r₁ + r₂ + ... + rₖ) / k.
3. Compute Context Precision as:
    Context Precision = (1/N) * (precision@1 + precision@2 + ... + precision@N).

Output:
Return the computed Context Precision as a single number between 0 and 1.
STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value. Stick to the precision calculation only.
""")

context_recall_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Context Recall metric based on the inputs provided below.

Inputs:
User Input: {user_input}
Reference: {reference}
Retrieved Contexts: {retrieved_contexts}

Definition of Context Recall:
Context Recall measures how many of the relevant documents or pieces of information were successfully retrieved. It focuses on not missing important results. A higher recall indicates that fewer important documents were left out. Since recall is about not missing anything, it always requires a reference for comparison. This metric uses the reference as a proxy for annotated reference contexts, which is useful because annotating reference contexts can be very time consuming.

Steps to Compute Context Recall:
1. Claim Extraction:
   - Break down the Reference into individual claims. Each claim represents a piece of important information that should ideally be supported by the retrieved contexts.
2. Attribution Analysis:
   - For each extracted claim, determine whether it is supported by one or more chunks in the Retrieved Contexts.
   - If a claim is supported, assign it a value of 1.
   - If a claim is not supported, assign it a value of 0.
3. Metric Calculation:
   - Let C be the total number of claims extracted from the Reference.
   - For each claim, let a_i be its attribution indicator (1 if supported, 0 if not).
   - Compute Context Recall as:
         Context Recall = (a1 + a2 + ... + aC) / C
   - This value will be between 0 and 1, where a higher value indicates that more important claims from the Reference are covered by the Retrieved Contexts.

Your Task:
Using the provided inputs, please:
- Break down the Reference into individual claims.
- For each claim, determine if it is supported by the Retrieved Contexts.
- Compute the Context Recall metric as described.

Output:
Return the final Context Recall value as a single number between 0 and 1.
STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value. Stick to the recall calculation only.
""")

context_entities_recall_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Context Entities Recall metric based on the inputs provided below.

Inputs:
- Reference: {reference}
- Retrieved Contexts: {retrieved_contexts}

Definition of Context Entities Recall:
Context Entities Recall measures the fraction of important entities present in the Reference that are successfully retrieved in the Retrieved Contexts. This metric is particularly useful in fact-based use cases (e.g., tourism help desks or historical QA) where the presence of specific entities is crucial.

How to Compute Context Entities Recall:
1. Entity Extraction:
   - Extract the set of entities from the Reference; denote this set as E_ref.
   - Extract the set of entities from the Retrieved Contexts; denote this set as E_retrieved.
2. Intersection Calculation:
   - Determine the common entities between E_ref and E_retrieved; denote this set as E_common.
3. Metric Calculation:
   - Compute Context Entities Recall as:
         (Number of entities in E_common) divided by (Number of entities in E_ref).

Output:
Return the Context Entities Recall as a single value between 0 and 1. A higher value indicates that more entities from the Reference are present in the Retrieved Contexts.

Example Walkthrough:
- Reference:
  "The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal."
  Extracted Entities: ['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal']

- High Entity Recall Retrieved Context:
  "The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it."
  Extracted Entities: ['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India']

- Low Entity Recall Retrieved Context:
  "The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination."
  Extracted Entities: ['Taj Mahal', 'UNESCO', 'India']

Your Task:
Using the provided inputs, please:
- Extract entities from the Reference and the Retrieved Contexts.
- Compute the intersection of these entity sets.
- Calculate the Context Entities Recall metric as described.
- Return the final metric as a value between 0 and 1.

STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value.
""")

noise_sensitivity_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Noise Sensitivity metric based on the inputs provided below.

Inputs:
- User Input: {user_input}
- Reference (Ground Truth): {reference}
- Generated Response: {response}
- Retrieved Contexts: {retrieved_contexts}

Definition of Noise Sensitivity:
Noise Sensitivity measures how often the system introduces errors by providing claims in its generated response that are incorrect—that is, not supported by the ground truth or the relevant retrieved contexts. This metric ranges from 0 to 1, where lower values indicate better performance (fewer errors).

How to Compute Noise Sensitivity:
1. Claim Extraction:
   - Break down the Generated Response into individual claims. Each claim should represent a distinct piece of information or assertion.
2. Relevance Verification:
   - For each extracted claim, determine whether it is correct by verifying:
       * It is supported by the Reference (ground truth).
       * It can be attributed to evidence in the Retrieved Contexts.
   - Label each claim as:
       * Correct (supported) if it aligns with the ground truth and is inferable from the retrieved contexts.
       * Incorrect (noise) if it is not supported by the ground truth or introduces extraneous information.
3. Metric Calculation:
   - Let T be the total number of claims extracted from the response.
   - Let N be the number of incorrect claims.
   - Compute Noise Sensitivity as:
         Noise Sensitivity = N / T
   - The final score will be a value between 0 and 1.

Example Walkthrough:
- Question: What is the Life Insurance Corporation of India (LIC) known for?
- Ground Truth (Reference): The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.
- Retrieved Contexts:
    * Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
    * Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
    * Context 3: As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.
- Generated Response: The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.
- Analysis:
    * Suppose the response contains 3 claims.
    * The claim "LIC contributes to the financial stability of the country" is not supported by the ground truth.
    * Thus, N = 1 (incorrect claim) and T = 3 (total claims).
    * Noise Sensitivity = 1 / 3 ≈ 0.333

Your Task:
Using the provided inputs, please:
- Extract individual claims from the Generated Response.
- For each claim, determine whether it is correct (i.e., supported by the Reference and the Retrieved Contexts) or incorrect.
- Count the total number of claims (T) and the number of incorrect claims (N).
- Calculate the Noise Sensitivity score as N/T.
- Return the final Noise Sensitivity score as a single value between 0 and 1.

STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value.
""")

response_relevancy_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Response Relevancy metric based on the inputs provided below.

Inputs:
- User Input: {user_input}
- Generated Response: {response}

Definition of Response Relevancy:
Response Relevancy measures how well the generated response aligns with the user input. An answer is considered relevant if it directly and appropriately addresses the original question. This metric does not evaluate factual accuracy but focuses on ensuring that the response captures the intent of the user input without being incomplete or including extraneous details.

How to Compute Response Relevancy:
1. Generate Artificial Questions:
   - Based on the Generated Response, generate a set of artificial questions that capture its key points.
   - Generate 3 questions by default. These questions should reflect the content and nuances of the response.
2. Compute Cosine Similarity:
   - For each generated question, compute its embedding.
   - Compute the embedding of the User Input.
   - Calculate the cosine similarity between the embedding of each generated question and that of the User Input.
3. Calculate the Average Similarity:
   - Take the average of the cosine similarity scores obtained for all generated questions.
   - This average score is the Response Relevancy metric.

Note:
- Although cosine similarity mathematically ranges from -1 to 1, in this context the scores typically fall between 0 and 1.
- A higher score indicates better alignment between the generated response and the user input.

Example Walkthrough:
- User Input: "Where is France and what is its capital?"
- Low Relevance Answer: "France is in western Europe."
- High Relevance Answer: "France is in western Europe and Paris is its capital."

For the low relevance answer, the artificial questions generated might not capture the full intent of the original query, resulting in lower cosine similarity scores. In contrast, a high relevancy answer would enable the generation of questions that closely mirror the original query, yielding a higher average cosine similarity.

Your Task:
Using the provided inputs, please:
- Generate 3 artificial questions based on the Generated Response.
- For each generated question, compute the cosine similarity with the User Input embedding.
- Calculate the average of these cosine similarity scores.
- Return the final Response Relevancy score.

STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value.
""")

faithfulness_template = ("""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Faithfulness metric based on the inputs provided below.

Inputs:
- User Input (Question): {user_input}
- Retrieved Contexts: {retrieved_contexts}
- Generated Response: {response}

Definition of Faithfulness:
Faithfulness measures how factually consistent the generated response is with the information provided in the retrieved contexts. A response is considered faithful if every claim (statement) it makes can be directly supported or inferred from the retrieved contexts. The metric ranges from 0 to 1, where higher scores indicate better factual consistency.

How to Compute Faithfulness:
1. Claim Extraction:
   - Break down the Generated Response into individual claims or statements.
   - Each claim should be a distinct piece of factual information extracted from the response.
2. Verification Against Retrieved Contexts:
   - For each extracted claim, verify whether it is supported or can be inferred from the Retrieved Contexts.
   - Label each claim as:
       * Supported, if the claim is consistent with and inferable from the retrieved contexts.
       * Not Supported, if the claim is inconsistent or not found in the retrieved contexts.
3. Metric Calculation:
   - Let T be the total number of claims extracted from the response.
   - Let S be the number of claims that are supported by the retrieved contexts.
   - Compute the Faithfulness score as: Faithfulness = S / T.
   - The final score will be a value between 0 and 1.

Example Walkthrough:
- User Input (Question): Where and when was Einstein born?
- Retrieved Context:
    "Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time."
- Generated Response:
    "Einstein was born in Germany on 20th March 1879."
Steps:
1. Extract Claims:
    - Claim 1: "Einstein was born in Germany."
    - Claim 2: "Einstein was born on 20th March 1879."
2. Verify Each Claim:
    - Claim 1 is supported by the retrieved context.
    - Claim 2 is not supported because the context states the birth date is 14th March 1879.
3. Calculate Faithfulness:
    - Total claims, T = 2.
    - Supported claims, S = 1.
    - Faithfulness Score: 1 / 2 = 0.5.

Your Task:
Using the provided inputs:
- Extract the individual claims from the Generated Response.
- For each claim, determine if it is supported by the Retrieved Contexts.
- Compute the Faithfulness score as (Number of Supported Claims) / (Total Number of Claims).
- Return the final Faithfulness score as a value between 0 and 1.
- STRICTLY output only the value. NO additional information or formatting is required.
You are FORBIDDEN from explaining your value.
""")

# Configuration
MODELS = [
    "mistral:7b-instruct-q4_0"
]
#   "qwen2.5:7b-instruct-q4_0"
#   "gemma2:9b-instruct-q4_0",
#   "phi3.5:3.8b-mini-instruct-q4_0",
#   "mistral:7b-instruct-q4_0",
#"deepseek-r1:7b-qwen-distill-q4_K_M",
#"deepseek-r1:8b-llama-distill-q4_K_M"
#"llama3.1:8b-instruct-q4_0"

CSV_FILES = [
    "Results_mistral_7b-instruct-q4_0.csv"
]
#     "Results_lly_InternLM3-8B-Instruct_8b-instruct-q4_0.csv",
#       "Results_mistral_7b-instruct-q4_0.csv"
#       "Results_phi3.5_3.8b-mini-instruct-q4_0.csv",
#       "Results_gemma2_9b-instruct-q4_0.csv",
#       "Results_qwen2.5_7b-instruct-q4_0.csv",
#       "Results_llama3.1_8b-instruct-q4_0.csv"

def extract_page_contents(text: str) -> list:
    """
    Extracts the content of each page from a string containing one or more 
    Document(...) objects. This function looks for the pattern:
      page_content="...text..."
    and returns a list where each element is the text for one page.
    """
    # This pattern captures everything between page_content=" and the next "
    pattern = r'page_content="(.*?)"'
    matches = re.findall(pattern, text, flags=re.DOTALL)
    # Return each captured page content as a single, stripped string
    return [match.strip() for match in matches if match.strip()]

def preprocess_dataset(df: pd.DataFrame):
    """
    Prepares dataset for evaluation by renaming columns and extracting
    the page_content from the retrieved contexts into a list of strings,
    where each element corresponds to one page's content.
    """
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    # For each row in 'retrieved_contexts', if it is a string, extract the page contents.
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: extract_page_contents(x) if isinstance(x, str) else x
    )
    
    # Each row now has retrieved_contexts as a list of strings,
    # where the first element is the first page context, the second is the second, etc.
    return processed_df



noise_sensitivity_prompt = PromptTemplate(
    input_variables=["user_input", "reference", "response", "retrieved_contexts"],
    template=noise_sensitivity_template
)

faithfulness_prompt = PromptTemplate(
    input_variables=["user_input", "response", "retrieved_contexts"],
    template=faithfulness_template
)

response_relevancy_prompt = PromptTemplate(
    input_variables=["user_input", "response"],
    template=response_relevancy_template
)

context_entities_recall_prompt = PromptTemplate(
      input_variables=["reference", "retrieved_contexts"],
      template=context_entities_recall_template
   )

context_recall_prompt = PromptTemplate(
      input_variables=["user_input", "reference", "retrieved_contexts"],
      template=context_recall_template
   )

context_precision_prompt = PromptTemplate(
      input_variables=["user_input", "reference", "retrieved_contexts"],
      template=context_precision_template
   )

def get_noise_sensitivity(
    llm,
    user_input: str,
    response: str,
    reference: str,
    retrieved_contexts: list
) -> Dict[str, Any]:
    """
    Runs user input + context + reference + response through noise_sensitivity prompt.
    """
    # Build the pipeline
    chain = noise_sensitivity_prompt | llm | StrOutputParser()
    
    noise_result = chain.invoke({
        "user_input": user_input,
        "reference": reference,
        "response": response,
        "retrieved_contexts": retrieved_contexts
    })
    
    # Return in a consistent dictionary format
    return {
        "Noise Sensitivity": noise_result
    }

def get_faithfulness(llm, user_input: str, response: str, retrieved_contexts: list) -> Dict[str, Any]:
    chain = faithfulness_prompt | llm | StrOutputParser()
    faith_result = chain.invoke({
        "user_input": user_input,
        "response": response,
        "retrieved_contexts": retrieved_contexts
    })
    return {
        "Faithfulness": faith_result
    }

def get_response_relevancy(llm, user_input: str, response: str) -> Dict[str, Any]:
    chain = response_relevancy_prompt | llm | StrOutputParser()
    relevancy_result = chain.invoke({
        "user_input": user_input,
        "response": response
    })
    return {
        "Response Relevancy": relevancy_result
    }

def get_context_entities_recall(llm, reference: str, retrieved_contexts: list) -> Dict[str, Any]:  
      chain = context_entities_recall_prompt | llm | StrOutputParser()
      entities_recall_result = chain.invoke({
         "reference": reference,
         "retrieved_contexts": retrieved_contexts
      })
      return {
         "Context Entities Recall": entities_recall_result
      }

def get_context_recall(llm, user_input: str, reference: str, retrieved_contexts: list) -> Dict[str, Any]:  
      chain = context_recall_prompt | llm | StrOutputParser()
      context_recall_result = chain.invoke({
         "user_input": user_input,
         "reference": reference,
         "retrieved_contexts": retrieved_contexts
      })
      return {
         "Context Recall": context_recall_result
      }

def get_context_precision(llm, user_input: str, reference: str, retrieved_contexts: list) -> Dict[str, Any]:
      chain = context_precision_prompt | llm | StrOutputParser()
      context_precision_result = chain.invoke({
         "user_input": user_input,
         "reference": reference,
         "retrieved_contexts": retrieved_contexts
      })
      return {
         "Context Precision": context_precision_result
      }


def evaluate_metrics_for_row(row: pd.Series, llm) -> Dict[str, Any]:
    """
    For a single row, gather user_input / response / reference / retrieved_contexts
    and invoke each metric function, returning a combined dictionary.
    """
    user_input = row.get("user_input", "")
    response = row.get("response", "")
    reference = row.get("reference", "")
    retrieved_contexts = row.get("retrieved_contexts", [])

    # Call each metric function
    noise = get_noise_sensitivity(llm, user_input, response, reference, retrieved_contexts)
    faith = get_faithfulness(llm, user_input, response, retrieved_contexts)
    relevancy = get_response_relevancy(llm, user_input, response)
    entities_recall = get_context_entities_recall(llm, reference, retrieved_contexts)
    context_recall = get_context_recall(llm, user_input, reference, retrieved_contexts)
    context_precision = get_context_precision(llm, user_input, reference, retrieved_contexts)
    
    # Combine everything
    results = {}
    results.update(noise)
    results.update(faith)
    results.update(relevancy)
    results.update(entities_recall)
    results.update(context_recall)
    results.update(context_precision)
    
    return results

def evaluate_dataset():
    """Process each CSV file, then evaluate all models for that CSV."""
    # Wrap CSV_FILES loop with tqdm
    for csv_file in tqdm(CSV_FILES, desc="Processing CSVs"):
        tqdm.write(f"\nProcessing CSV: {csv_file}")  # Use tqdm.write for messages
        df = pd.read_csv(csv_file)
        processed_df = preprocess_dataset(df)
        
        # Wrap MODELS loop with tqdm
        for model_name in tqdm(MODELS, desc="Models", leave=False):
            tqdm.write(f"Evaluating {model_name} on {csv_file}")
            llm = ChatOllama(model=model_name, temperature=0, num_ctx=20000)
            model_evals = []
            
            # Wrap row iteration with tqdm
            rows = processed_df.iterrows()
            for idx, row in tqdm(rows, total=len(processed_df), desc="Rows", leave=False):
                row_results = evaluate_metrics_for_row(row, llm)
                row_results["row_index"] = idx
                model_evals.append(row_results)
            
            # Generate output filename
            csv_base = Path(csv_file).stem
            safe_model_name = model_name.replace(":", "_").replace("/", "_")
            output_filename = f"{csv_base}_EvaluatedBy_{safe_model_name}.csv"
            pd.DataFrame(model_evals).to_csv(output_filename, index=False)
            tqdm.write(f"Saved: {output_filename}")

if __name__ == "__main__":
    evaluate_dataset()


