
import pandas as pd
from detoxify import Detoxify

# Load the CSV file
df = pd.read_csv('data.csv')

# Initialize the Detoxify model
model = Detoxify('unbiased')

# Function to classify a single answer
def classify_answer(answer):
    results = model.predict(answer)
    return results

# Apply the classification to each entry in the 'answer' column
classification_results = df['answer'].apply(classify_answer)

# Create new columns for each toxicity label
df['toxicity'] = classification_results.apply(lambda x: x['toxicity'])
df['severe_toxicity'] = classification_results.apply(lambda x: x['severe_toxicity'])
df['obscene'] = classification_results.apply(lambda x: x['obscene'])
df['identity_attack'] = classification_results.apply(lambda x: x['identity_attack'])
df['insult'] = classification_results.apply(lambda x: x['insult'])
df['threat'] = classification_results.apply(lambda x: x['threat'])
df['sexual_explicit'] = classification_results.apply(lambda x: x['sexual_explicit'])

# Save the results to a new CSV file
df.to_csv('abc.csv', index=False)

