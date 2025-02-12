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


# Configuration
MODELS = [
    "lly/InternLM3-8B-Instruct:8b-instruct-q4_0",
    "llama3.1:8b-instruct-q4_0",
    "qwen2.5:7b-instruct-q4_0",
    "gemma2:9b-instruct-q4_0",
    "phi3.5:3.8b-mini-instruct-q4_0",
    "mistral:7b-instruct-q4_0",
    "deepseek-r1:7b-qwen-distill-q4_K_M",
    "deepseek-r1:8b-llama-distill-q4_K_M"
]

CSV_FILES = [
    "Results_lly_InternLM3-8B-Instruct:8b-instruct-q4_0.csv",
    "Results_mistral:7b-instruct-q4_0.csv",
    "Results_phi3.5:3.8b-mini-instruct-q4_0.csv",
    "Results_gemma2:9b-instruct-q4_0.csv",
    "Results_qwen2.5:7b-instruct-q4_0.csv",
    "Results_llama3.1:8b-instruct-q4_0.csv"
]

def preprocess_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataset for evaluation"""
    processed_df = df.rename(columns={
        "Question": "user_input",
        "Context": "retrieved_contexts",
        "Answer": "response",
        "Ground_Truth": "reference"
    })
    
    # Convert string representation of list to actual list
    processed_df['retrieved_contexts'] = processed_df['retrieved_contexts'].apply(
        lambda x: ast.literal_eval(x) if isinstance(x, str) else x
    )
    return processed_df



context_precision_template=(r""" You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Context Precision metric based on the inputs provided below. Inputs:

User Input: {user_input}
Reference: {reference}
Retrieved Contexts: {retrieved_contexts}

Definition of Context Precision: Context Precision measures the proportion of relevant chunks within the retrieved contexts. It is computed as the mean of the precision@k values for each retrieved context chunk. For a list of retrieved context chunks:

Let N be the total number of chunks in retrieved_contexts.
For each rank k (1 ≤ k ≤ N), define:
precision@k = (Number of relevant chunks in the top k) / k.
A chunk is considered relevant if it provides information that aligns with the Reference (and supports answering the User Input).
For each chunk at rank k, let the relevance indicator rₖ = 1 if the chunk is relevant, and 0 if it is not.
Then, Context Precision is given by:
Context Precision = (1/N) * ∑ₖ₌₁ᴺ precision@k

The final output should be a value between 0 and 1. Your Task:

Evaluate Relevance:
For each chunk in retrieved_contexts, determine whether it is relevant to the Reference (and consider the User Input if needed). Mark it with a relevance indicator rₖ (1 for relevant, 0 for not).
Compute Precision@k:
For each rank k from 1 to N, calculate: [ \text{{precision@k}} = \frac{{\sum_{{i=1}}^{{k}} r_i}}{{k}} ]
Calculate Context Precision:
Compute the mean of all the precision@k values: [ \text{{Context Precision}} = \frac{{1}}{{N}} \sum_{{k=1}}^{{N}} \text{{precision@k}} ]
Output:
Return the computed Context Precision as a single number between 0 and 1.

Please perform these calculations and provide the final Context Precision value. STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value. Stick to the precision calculation only. """ 
)

context_recall_template=(r""" You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the Context Recall metric based on the inputs provided below. Inputs:

User Input: {user_input}
Reference: {reference}
Retrieved Contexts: {retrieved_contexts}

Definition of Context Recall: Context Recall measures how many of the relevant documents (or pieces of information) were successfully retrieved. It focuses on not missing important results. Higher recall indicates that fewer important documents were left out. Since recall is about not missing anything, it always requires a reference for comparison. This metric uses the reference as a proxy for annotated reference contexts, which is useful because annotating reference contexts can be very time consuming. Steps to Compute Context Recall:

Claim Extraction:
Break down the Reference into individual claims. Each claim represents a piece of important information that should ideally be supported by the retrieved contexts.
Attribution Analysis:
For each extracted claim, analyze whether it is supported (attributed) to one or more chunks in Retrieved Contexts.
If a claim is supported by the retrieved context, assign it a value of 1.
If a claim is not supported, assign it a value of 0.
Metric Calculation:
Let CC be the total number of claims extracted from the reference. For each claim ii (where i=1,2,...,Ci=1,2,...,C), let aiai​ be its attribution indicator (1 if supported, 0 otherwise).
The Context Recall is calculated as: Context Recall=∑i=1CaiCContext Recall=C∑i=1C​ai​​ This value will be between 0 and 1, where a higher value indicates that more important claims from the reference are covered by the retrieved contexts.

Your Task: Using the provided inputs, please:

Break down the reference into individual claims.
For each claim, determine if it is supported by the retrieved contexts.
Compute the Context Recall metric as described.
Return the final Context Recall value as a single number between 0 and 1. STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value. Stick to the recall calculation only. """ 
)


context_entities_recall_template=(r"""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the **Context Entities Recall** metric based on the inputs provided below.

**Inputs:**
- **Reference:** {reference}
- **Retrieved Contexts:** {retrieved_contexts}

**Definition of Context Entities Recall:**

The Context Entities Recall metric measures the fraction of important entities present in the reference that are successfully retrieved in the retrieved contexts. This is particularly useful in fact-based use cases (e.g., tourism help desks or historical QA) where the presence of specific entities is crucial.

**How to Compute Context Entities Recall:**

1. **Entity Extraction:**  
- Extract the set of entities from the **Reference**. Denote this set as \( E_{{ref}} \).  
- Extract the set of entities from the **Retrieved Contexts**. Denote this set as \( E_{{retrieved}} \).

2. **Intersection Calculation:**  
- Determine the common entities between \( E_{{ref}} \) and \( E_{{retrieved}} \), i.e.,  
\[
E_{{common}} = E_{{ref}} \cap E_{{retrieved}}
\]

3. **Metric Calculation:**  
- Calculate the Context Entities Recall using the formula:
\[
\text{Context Entities Recall} = \frac{|E_{common}|}{|E_{ref}|}
\]
where \( |E_{common}| \) is the number of entities common to both sets, and \( |E_{ref}| \) is the total number of entities in the reference.

4. **Output:**  
- Return the Context Entities Recall as a single value between 0 and 1. A higher value indicates that more entities from the reference are present in the retrieved contexts.

**Example Walkthrough:**

- **Reference:**  
"The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in the Indian city of Agra. It was commissioned in 1631 by the Mughal emperor Shah Jahan to house the tomb of his favorite wife, Mumtaz Mahal."  
*Extracted Entities:* \['Taj Mahal', 'Yamuna', 'Agra', '1631', 'Shah Jahan', 'Mumtaz Mahal'\]

- **High Entity Recall Retrieved Context:**  
"The Taj Mahal is a symbol of love and architectural marvel located in Agra, India. It was built by the Mughal emperor Shah Jahan in memory of his beloved wife, Mumtaz Mahal. The structure is renowned for its intricate marble work and beautiful gardens surrounding it."  
*Extracted Entities:* \['Taj Mahal', 'Agra', 'Shah Jahan', 'Mumtaz Mahal', 'India'\]

- **Low Entity Recall Retrieved Context:**  
"The Taj Mahal is an iconic monument in India. It is a UNESCO World Heritage Site and attracts millions of visitors annually. The intricate carvings and stunning architecture make it a must-visit destination."  
*Extracted Entities:* \['Taj Mahal', 'UNESCO', 'India'\]

In this example, the first retrieved context would have a higher Context Entities Recall compared to the second if it covers more of the entities in the reference.

**Your Task:**

Using the provided inputs, please:
- Extract entities from the **Reference** and the **Retrieved Contexts**.
- Compute the intersection of these entity sets.
- Calculate the Context Entities Recall using the formula above.
- Return the final metric as a value between 0 and 1.

Please compute the **Context Entities Recall** metric. STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value.
"""
)

noise_sensitivity_template=(r"""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the **Noise Sensitivity** metric based on the inputs provided below.

**Inputs:**
- **User Input:** {user_input}
- **Reference (Ground Truth):** {reference}
- **Generated Response:** {response}
- **Retrieved Contexts:** {retrieved_contexts}

**Definition of Noise Sensitivity:**

Noise Sensitivity measures how often the system introduces errors by providing claims in its generated response that are incorrect—i.e., not supported by the ground truth or the relevant retrieved context. This metric ranges from 0 to 1, where lower values indicate better performance (fewer errors).

**How to Compute Noise Sensitivity:**

1. **Claim Extraction:**  
- Break down the generated **Response** into individual claims. Each claim should represent a distinct piece of information or assertion.

2. **Relevance Verification:**  
- For each extracted claim, determine whether it is correct by verifying:
- It is supported by the **Reference** (ground truth).
- It can be attributed to the evidence in the **Retrieved Contexts**.
- Label each claim as:
- **Correct (supported)** if it aligns with the ground truth and is inferable from the retrieved contexts.
- **Incorrect (noise)** if it is not supported by the ground truth or introduces extraneous information.

3. **Metric Calculation:**  
- Let \(T\) be the total number of claims extracted from the response.
- Let \(N\) be the number of incorrect claims (noise).
- Compute the **Noise Sensitivity** as:
\[
\text{{Noise Sensitivity}} = \frac{{N}}{{T}}
\]
- The final score will be a value between 0 and 1.

**Example Walkthrough:**

- **Question:**  
What is the Life Insurance Corporation of India (LIC) known for?

- **Ground Truth (Reference):**  
The Life Insurance Corporation of India (LIC) is the largest insurance company in India, established in 1956 through the nationalization of the insurance industry. It is known for managing a large portfolio of investments.

- **Relevant Retrieved Contexts:**  
- Context 1: The Life Insurance Corporation of India (LIC) was established in 1956 following the nationalization of the insurance industry in India.
- Context 2: LIC is the largest insurance company in India, with a vast network of policyholders and a significant role in the financial sector.
- Context 3: As the largest institutional investor in India, LIC manages substantial funds, contributing to the financial stability of the country.

- **Generated Response:**  
The Life Insurance Corporation of India (LIC) is the largest insurance company in India, known for its vast portfolio of investments. LIC contributes to the financial stability of the country.

*Analysis:*  
- Suppose the response contains 3 claims.
- The claim "LIC contributes to the financial stability of the country" is not supported by the ground truth.
- Thus, \( N = 1 \) (incorrect claim) and \( T = 3 \) (total claims).

**Noise Sensitivity** = \( \frac{{1}}{{3}} \approx 0.333 \)

**Your Task:**

Using the provided inputs:
- Extract individual claims from the generated **Response**.
- For each claim, determine whether it is correct (i.e., supported by the **Reference** and the **Retrieved Contexts**) or incorrect.
- Count the total number of claims \(T\) and the number of incorrect claims \(N\).
- Calculate the Noise Sensitivity score as \( N/T \).
- Return the final Noise Sensitivity score as a single value between 0 and 1.

Please compute and provide the **Noise Sensitivity** metric. STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value.
"""
)

response_relevancy_template=(r"""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the **Response Relevancy** metric based on the inputs provided below.

**Inputs:**
- **User Input:** {user_input}
- **Generated Response:** {response}

**Definition of Response Relevancy:**

Response Relevancy measures how well the generated response aligns with the user input. An answer is considered relevant if it directly and appropriately addresses the original question. This metric does not evaluate factual accuracy, but rather focuses on ensuring that the response captures the intent of the user input without being incomplete or including extraneous details.

**How to Compute Response Relevancy:**

1. **Generate Artificial Questions:**  
- Using the content of the **Generated Response**, generate a set of artificial questions that capture its key points.
- Generate **n = 3** questions by default. These questions should be crafted to reflect the content and nuances of the response.

2. **Compute Cosine Similarity:**  
- For each generated question, compute its embedding.
- Compute the embedding of the **User Input**.
- Calculate the cosine similarity between the embedding of each generated question and the embedding of the user input.

3. **Calculate the Average Similarity:**  
- Take the average of the cosine similarity scores obtained for all generated questions.
- This average score represents the **Response Relevancy** metric.

**Formula:**

Let:
- \( Q_i \) be the embedding of the i-th generated question.
- \( E_{{user}} \) be the embedding of the user input.
- \( n \) be the number of generated questions (default is 3).

Then:
\[
\text{{Response Relevancy}} = \frac{{1}}{{n}} \sum_{{i=1}}^{{n}} \text{{cosine\_similarity}}(Q_i, E_{{user}})
\]

**Note:**  
- Although cosine similarity mathematically ranges from -1 to 1, in this context the scores typically fall between 0 and 1.
- A higher score indicates better alignment between the generated response and the user input.

**Example Walkthrough:**

- **User Input:** "Where is France and what is its capital?"
- **Low Relevance Answer:** "France is in western Europe."
- **High Relevance Answer:** "France is in western Europe and Paris is its capital."

For the low relevance answer, the artificial questions generated might not capture the full intent of the original query, resulting in lower cosine similarity scores. In contrast, a high relevancy answer would enable the generation of questions that closely mirror the original query, yielding a higher average cosine similarity.

**Your Task:**

Using the provided inputs:
- Generate 3 artificial questions based on the **Generated Response**.
- For each generated question, compute the cosine similarity with the **User Input** embedding.
- Calculate the average of these cosine similarity scores.
- Return the final **Response Relevancy** score.

Please compute and provide the **Response Relevancy** metric. STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value.
"""
)

faithfulness_template=(r"""
You are a helpful assistant tasked with evaluating the performance of a Retrieval-Augmented Generation (RAG) system. Your job is to compute the **Faithfulness** metric based on the inputs provided below.

**Inputs:**
- **User Input (Question):** {user_input}
- **Retrieved Contexts:** {retrieved_contexts}
- **Generated Response:** {response}

**Definition of Faithfulness:**

Faithfulness measures how factually consistent the generated response is with the information provided in the retrieved contexts. A response is considered faithful if every claim (statement) it makes can be directly supported or inferred from the retrieved contexts. The metric ranges from 0 to 1, where higher scores indicate better factual consistency.

**How to Compute Faithfulness:**

1. **Claim Extraction:**  
- Break down the **Generated Response** into individual claims or statements.  
- Each claim should be a distinct piece of factual information extracted from the response.

2. **Verification Against Retrieved Contexts:**  
- For each extracted claim, verify whether it is supported or can be inferred from the **Retrieved Contexts**.
- Label each claim as:
- **Supported:** if the claim is consistent with and inferable from the retrieved contexts.
- **Not Supported:** if the claim is inconsistent or not found in the retrieved contexts.

3. **Metric Calculation:**  
- Let \( T \) be the total number of claims extracted from the response.
- Let \( S \) be the number of claims that are supported by the retrieved contexts.
- Compute the **Faithfulness** score using the formula:
\[
\text{{Faithfulness}} = \frac{{S}}{{T}}
\]
- The final score will be a value between 0 and 1.

**Example Walkthrough:**

- **User Input (Question):** Where and when was Einstein born?
- **Retrieved Context:**  
"Albert Einstein (born 14 March 1879) was a German-born theoretical physicist, widely held to be one of the greatest and most influential scientists of all time."
- **Generated Response:**  
"Einstein was born in Germany on 20th March 1879."

**Steps:**

1. **Extract Claims:**  
- Claim 1: "Einstein was born in Germany."  
- Claim 2: "Einstein was born on 20th March 1879."

2. **Verify Each Claim:**  
- Claim 1 is supported by the retrieved context.  
- Claim 2 is not supported because the context states the birth date is 14th March 1879.

3. **Calculate Faithfulness:**  
- Total claims, \( T = 2 \)  
- Supported claims, \( S = 1 \)  
- **Faithfulness Score:** \( \frac{{1}}{{2}} = 0.5 \)

**Your Task:**

Using the provided inputs:
- Extract the individual claims from the **Generated Response**.
- For each claim, determine if it is supported by the **Retrieved Contexts**.
- Compute the Faithfulness score as \( \frac{\text{Number of Supported Claims}}{\text{Total Number of Claims}} \).
- Return the final Faithfulness score as a value between 0 and 1.
- STRICTLY output only the value. NO additional information or formatting is required. You are FORBIDDEN from explaining your value.
"""
)

class MetricEvaluator:
    def __init__(self):
        self.metrics = self.initialize_metrics()
        self.output_parser = StrOutputParser()
        
    def initialize_metrics(self):
        """Define all evaluation metrics as separate chains"""
        return {
            'noise_sensitivity': PromptTemplate(
                input_variables=["user_input", "reference", "response", "retrieved_contexts"],
                template=noise_sensitivity_template
            ),
            'faithfulness': PromptTemplate(
                input_variables=["user_input", "response", "retrieved_contexts"],
                template=faithfulness_template
            ),
            'response_relevancy': PromptTemplate(
                input_variables=["user_input", "response"],
                template=response_relevancy_template
            ),
            'context_entities_recall': PromptTemplate(
                input_variables=["reference", "retrieved_contexts"],
                template=context_entities_recall_template
            ),
            'context_recall': PromptTemplate(
                input_variables=["user_input", "reference", "retrieved_contexts"],
                template=context_recall_template
            ),
            'context_precision': PromptTemplate(
                input_variables=["user_input", "reference", "retrieved_contexts"],
                template=context_precision_template
            )
        }
    
    def create_chains(self, model_name):
        """Create evaluation chains for all metrics using specified model"""
        llm = ChatOllama(model=model_name, temperature=0)
        return {
            metric_name: prompt | llm | self.output_parser
            for metric_name, prompt in self.metrics.items()
        }
    
    def preprocess_data(self, df):
        """Prepare dataset format"""
        df = df.rename(columns={
            "Question": "user_input",
            "Context": "retrieved_contexts",
            "Answer": "response",
            "Ground_Truth": "reference"
        })
        df['retrieved_contexts'] = df['retrieved_contexts'].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else x
        )
        return df
    
    def evaluate_row(self, row, chains):
        """Evaluate a single row with all metrics"""
        results = {}
        contexts = "\n".join([f"- {ctx}" for ctx in row['retrieved_contexts']])
        
        for metric_name, chain in chains.items():
            try:
                result = chain.invoke({
                    "user_input": row['user_input'],
                    "reference": row['reference'],
                    "response": row['response'],
                    "retrieved_contexts": contexts
                })
                results[metric_name] = float(result.strip())
            except Exception as e:
                print(f"Error in {metric_name}: {str(e)}")
                results[metric_name] = None
            time.sleep(1)  # Rate limiting
            
        return results
    
    def process_file(self, csv_path):
        """Process a CSV file with all models and metrics"""
        df = self.preprocess_data(pd.read_csv(csv_path))
        
        for model in MODELS:
            model_safe = model.replace('/', '_').replace(':', '_')
            print(f"\nEvaluating with {model}...")
            
            # Initialize chains for current model
            chains = self.create_chains(model)
            
            # Add columns if missing
            for metric in self.metrics:
                col_name = f"{metric}_{model_safe}"
                if col_name not in df.columns:
                    df[col_name] = None
                    
            # Process rows
            for idx, row in tqdm(df.iterrows(), total=len(df)):
                if any(pd.isnull(row[f"{metric}_{model_safe}"]) for metric in self.metrics):
                    results = self.evaluate_row(row, chains)
                    for metric, value in results.items():
                        df.at[idx, f"{metric}_{model_safe}"] = value
                        
            # Save incremental progress
            df.to_csv(csv_path.replace('.csv', '_partial.csv'), index=False)
            
        # Final save
        df.to_csv(csv_path.replace('.csv', '_evaluated.csv'), index=False)
        return df

    def test_system(self):
        """Test the evaluation system with sample data"""
        sample_row = {
            "user_input": "What is photosynthesis?",
            "reference": "Photosynthesis is the process plants use to convert sunlight into energy.",
            "response": "Plants create energy through photosynthesis using sunlight.",
            "retrieved_contexts": [
                "Photosynthesis converts light energy to chemical energy",
                "Occurs in chloroplasts of plant cells"
            ]
        }
        
        print("Running system test...")
        for model in MODELS:
            print(f"\nTesting {model}:")
            chains = self.create_chains(model)
            results = self.evaluate_row(sample_row, chains)
            for metric, value in results.items():
                print(f"{metric}: {value}")
            print("-------------------")

if __name__ == "__main__":
    evaluator = MetricEvaluator()
    
    # 1. Run system test
    evaluator.test_system()
    
    # 2. Process files after confirmation
    proceed = input("\nProceed with full evaluation? (y/n): ")
    if proceed.lower() == 'y':
        for csv_file in CSV_FILES:
            print(f"\n{'='*50}\nProcessing {csv_file}")
            evaluator.process_file(csv_file)