# Libraries
import pyterrier as pt
import datasets
import pandas as pd
from tqdm import tqdm
from micracl_bm25 import *
from sklearn.ensemble import RandomForestRegressor

import nltk

nltk.download("punkt")


#TODO: load dataset in a way that is compatible with PyTerrier

# Load miracl datasets
lang = 'sw'  # choose language
miracl_corpus_hf = datasets.load_dataset('miracl/miracl-corpus', lang, trust_remote_code=True)
miracl_queries_hf = datasets.load_dataset('miracl/miracl', lang, trust_remote_code=True)

# Convert Hugging Face datasets to PyTerrier dataset objects
miracl_corpus_pt = pt.datasets(miracl_corpus_hf['train'], columns=["id", "text"])
miracl_queries_pt = pt.dataset(miracl_queries_hf, columns=["id", "text"])

# Access the datasets as required
print(miracl_corpus_pt.get_corpus())
print(miracl_queries_pt.get_topics())


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

#TODO: implement first-stage retrieval

# Load the index
index = pt.IndexFactory.of(miracl_index_path)

# Get the first 100 results using BM25
get_100_bm25 = pt.BatchRetrieve(
    index,
    wmodel="BM25",
    num_results=100
)

#TODO: compute features

TF_IDF =  pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"})
PL2 =  pt.BatchRetrieve(index, controls = {"wmodel": "PL2"})

pipe = get_100_bm25 >> (TF_IDF ** PL2)

pipe_fast = pipe.compile()

fbr = pt.FeaturesBatchRetrieve(index, controls = {"wmodel": "BM25"}, features=["SAMPLE", "WMODEL:TF_IDF", "WMODEL:PL2"]) 

BaselineLTR = fbr >> pt.pipelines.LTR_pipeline(RandomForestRegressor(n_estimators=400))
BaselineLTR.fit(train_topics, qrels)

results = pt.pipelines.Experiment([PL2, BaselineLTR], test_topics, qrels, ["map"], names=["PL2 Baseline", "LTR Baseline"])
results

#TODO: train learning to rank models


