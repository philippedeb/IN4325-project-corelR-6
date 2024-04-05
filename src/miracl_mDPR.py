from transformers import AutoTokenizer, AutoModelForMaskedLM
from typing import Tuple
import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import datasets
import os
import json

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
LANGUAGES_FOLDER = os.path.join(DATA_FOLDER, 'languages')
LANGUAGES = json.load(open(os.path.join(DATA_FOLDER, 'languages.json')))

# Initialize PyTerrier
if not pt.started():
    pt.init()

# Load model directly: "mDPR" model
tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-multilingual-cased")
model = AutoModelForMaskedLM.from_pretrained("google-bert/bert-base-multilingual-cased")

def encode_documents(language: str, split: str = "dev", max_seq_length: int = 512) -> None:
    LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, language)
    if not os.path.exists(LANGUAGE_FOLDER):
        os.makedirs(LANGUAGE_FOLDER)
    ENCODINGS_FOLDER = os.path.join(LANGUAGE_FOLDER, 'encodings')
    if not os.path.exists(ENCODINGS_FOLDER):
        os.makedirs(ENCODINGS_FOLDER)
    ENCODING_SPLIT_FOLDER = os.path.join(ENCODINGS_FOLDER, split)
    if not os.path.exists(ENCODING_SPLIT_FOLDER):
        os.makedirs(ENCODING_SPLIT_FOLDER)
    ENCODING_JSON = os.path.join(ENCODING_SPLIT_FOLDER, 'document_encodings.json')
    
    corpora = datasets.load_dataset('miracl/miracl-corpus', language, trust_remote_code=True)
    assert split in corpora, f"Split {split} not found for language {language}"

    desc = f"Encoding {language} corpus ({split})"
    encodings = {}
    for doc in tqdm(corpora[split], desc=desc):
        doc_id = doc['doc_id']
        doc_title = doc['title']
        doc_text = doc['text']
        doc_encoding = tokenizer(doc_title + doc_text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        encodings[doc_id] = doc_encoding
    with open(ENCODING_JSON, 'w') as f:
        json.dump(encodings, f)
    

def encode_queries(language: str, split: str = "dev", max_seq_length: int = 512) -> None:
    LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, language)
    if not os.path.exists(LANGUAGE_FOLDER):
        os.makedirs(LANGUAGE_FOLDER)
    ENCODINGS_FOLDER = os.path.join(LANGUAGE_FOLDER, 'encodings')
    if not os.path.exists(ENCODINGS_FOLDER):
        os.makedirs(ENCODINGS_FOLDER)
    ENCODING_SPLIT_FOLDER = os.path.join(ENCODINGS_FOLDER, split)
    if not os.path.exists(ENCODING_SPLIT_FOLDER):
        os.makedirs(ENCODING_SPLIT_FOLDER)
    ENCODING_JSON = os.path.join(ENCODING_SPLIT_FOLDER, 'query_encodings.json')
    
    queries = datasets.load_dataset('miracl/miracl', language, trust_remote_code=True)
    assert split in queries, f"Split {split} not found for language {language}"

    desc = f"Encoding {language} queries ({split})"
    encodings = {}
    for query in tqdm(queries[split], desc=desc):
        query_id = query['query_id']
        query_text = query['text']
        query_encoding = tokenizer(query_text, truncation=True, padding='max_length', max_length=max_seq_length, return_tensors='pt')
        encodings[query_id] = query_encoding
    with open(ENCODING_JSON, 'w') as f:
        json.dump(encodings, f)

