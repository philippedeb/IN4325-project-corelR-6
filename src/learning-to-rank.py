# Libraries
import pyterrier as pt
import datasets
import pandas as pd
from tqdm import tqdm
from micracl_bm25 import *

import nltk

nltk.download("punkt")

# Load miracl datasets
lang='sw'   # choose language
miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', lang, trust_remote_code=True) # splits: train
miracl_queries = datasets.load_dataset('miracl/miracl', lang, trust_remote_code=True)       # splits: train, dev, testA, testB

# Set up the folder to store the results
LANGUAGES_FOLDER = os.path.join(DATA_FOLDER, 'languages')
LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, lang)

INDICES_FOLDER = os.path.join(LANGUAGE_FOLDER, 'indices')
INDEX_SPLIT_FOLDER = os.path.join(INDICES_FOLDER, "train")  # Only a "train" split is available, = everything
miracl_index_path = os.path.join(INDEX_SPLIT_FOLDER, 'miracl_index')

# If miracl_index does not exist, create it
if not os.path.exists(miracl_index_path):
	index_miracl_corpus(language)

# Change the data.properties file to use in-memory data and indices
DATA_PROPERTIES = os.path.join(miracl_index_path, 'data.properties')

with open(DATA_PROPERTIES, 'r') as f:
	lines = f.readlines()
	
with open(DATA_PROPERTIES, 'w') as f:
	for line in lines:
		if 'index.meta.data-source=file' in line:
			f.write('index.meta.data-source=fileinmem\n')
		elif 'index.meta.index-source=file' in line:
			f.write('index.meta.index-source=fileinmem\n')
		else:
			f.write(line)

# Load the index
index = pt.IndexFactory.of(miracl_index_path)

get_100_bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    num_results=100
)
