import pandas as pd
import numpy as np
results_set = pd.read_csv('Results_AdvancedRAG.csv')
test_set = pd.read_csv('ROBIN_FINAL_TEST_SET.csv')
import re
# Define a function to extract context and answer using regular expressions
def extract_context_answer(text):
    match = re.search(r'CONTEXT:(.*?)QUESTION:(.*?)INST(.*)$', text, re.DOTALL)
    if match:
        context = match.group(1).strip()
        answer = match.group(3).strip().replace("\n", " ").replace("\r", "").replace("[/", "").replace("]", "")
        return context, answer
    else:
        return None, None
    

results_set[['context', 'answer']] = results_set['Answer'].apply(lambda x: pd.Series(extract_context_answer(x)))
results_set
results_set = results_set.drop(columns=['Answer'])
results_set
results_set = results_set.rename(columns={'Question': 'question'})
results_set
evaluation_set_Mistral7B = pd.merge(results_set, test_set, on='question')
evaluation_set_Mistral7B
evaluation_set_Mistral7B.drop_duplicates(inplace=True)
evaluation_set_Mistral7B.dropna(inplace=True)
evaluation_set_Mistral7B

evaluation_set_Mistral7B.to_csv('evaluation_set_Advanced.csv', index=False)