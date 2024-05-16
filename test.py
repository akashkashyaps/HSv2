# test_imports.py
import sys
print("Python executable:", sys.executable)

try:
    import torch
    print("torch imported successfully")
except ImportError as e:
    print(f"Error importing torch: {e}")

try:
    import transformers
    print("transformers imported successfully")
except ImportError as e:
    print(f"Error importing transformers: {e}")

try:
    import pandas as pd
    print("pandas imported successfully")
except ImportError as e:
    print(f"Error importing pandas: {e}")

try:
    import numpy as np
    print("numpy imported successfully")
except ImportError as e:
    print(f"Error importing numpy: {e}")

try:
    from torch import cuda, bfloat16
    print("torch.cuda and torch.bfloat16 imported successfully")
except ImportError as e:
    print(f"Error importing torch.cuda or torch.bfloat16: {e}")

try:
    from langchain.llms import HuggingFacePipeline
    print("HuggingFacePipeline imported successfully")
except ImportError as e:
    print(f"Error importing HuggingFacePipeline: {e}")

try:
    from langchain.document_loaders.csv_loader import CSVLoader
    print("CSVLoader imported successfully")
except ImportError as e:
    print(f"Error importing CSVLoader: {e}")

try:
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    print("RecursiveCharacterTextSplitter imported successfully")
except ImportError as e:
    print(f"Error importing RecursiveCharacterTextSplitter: {e}")

try:
    from langchain.embeddings import HuggingFaceEmbeddings
    print("HuggingFaceEmbeddings imported successfully")
except ImportError as e:
    print(f"Error importing HuggingFaceEmbeddings: {e}")

try:
    from langchain.vectorstores import Chroma
    print("Chroma imported successfully")
except ImportError as e:
    print(f"Error importing Chroma: {e}")

try:
    from langchain_community.document_loaders.csv_loader import CSVLoader as CSVLoaderCommunity
    print("CSVLoader from langchain_community imported successfully")
except ImportError as e:
    print(f"Error importing CSVLoader from langchain_community: {e}")

try:
    from langchain.chains import RetrievalQA
    print("RetrievalQA imported successfully")
except ImportError as e:
    print(f"Error importing RetrievalQA: {e}")

try:
    from transformers import StoppingCriteriaList, StoppingCriteria
    print("StoppingCriteriaList and StoppingCriteria imported successfully")
except ImportError as e:
    print(f"Error importing StoppingCriteriaList or StoppingCriteria: {e}")

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print("AutoTokenizer and AutoModelForCausalLM imported successfully")
except ImportError as e:
    print(f"Error importing AutoTokenizer or AutoModelForCausalLM: {e}")
