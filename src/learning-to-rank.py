# Libraries
import pyterrier as pt
import datasets
import pandas as pd
import numpy as np
from tqdm import tqdm
from micracl_bm25 import *
from sklearn.ensemble import RandomForestRegressor


def get_dataframe_qrels(language: str, split: str = "dev") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a dataframe containing the qrels of a specific language.

    Args:
        language (str): string representing the language of the dataset to load.
        split (str): string representing the split of the dataset to load. Default is "dev".

    Returns:
        DataFrame: a dataframe containing the qrels of a specific language.
    """
    assert language in queries, f"Language {language} not loaded"
    assert split in queries[language], f"Split {split} not found for language {language}"

    # Preparing qrels for PyTerrier
    qrels_pyt = []

    for idx, data in enumerate(tqdm(queries[language][split], desc="Processing Qrels")):
        for entry in data['positive_passages']:
            qrels_pyt.append({'qid': data['query_id'],
                        'docno': entry['docid'], 'label': 1})
        for entry in data['negative_passages']:
            qrels_pyt.append({'qid': data['query_id'],
                        'docno': entry['docid'], 'label': 0})

    qrels_df = pd.DataFrame(qrels_pyt)

    return qrels_df

# Load files and index

language='sw' # Choose language
split = 'dev' # Set split
load_data_language(language)

# Load the index
LANGUAGES_FOLDER = os.path.join(DATA_FOLDER, 'languages')
LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, language)
BM25_FOLDER = os.path.join(LANGUAGE_FOLDER, 'bm25')
INDICES_FOLDER = os.path.join(LANGUAGE_FOLDER, 'indices')
INDEX_SPLIT_FOLDER = os.path.join(INDICES_FOLDER, "train")
miracl_index_path = os.path.join(INDEX_SPLIT_FOLDER, 'miracl_index')
index = pt.IndexFactory.of(miracl_index_path)

# Load the qrels
qrels = get_dataframe_qrels(language)

# Multi-stage retrieval

#this ranker will make the candidate set of documents for each query
BM25 = pt.BatchRetrieve(index, controls = {"wmodel": "BM25"}, num_results = 100)

#these rankers we will use to re-rank the BM25 results
TF_IDF =  pt.BatchRetrieve(index, controls = {"wmodel": "TF_IDF"})
PL2 =  pt.BatchRetrieve(index, controls = {"wmodel": "PL2"})

pipe = BM25 >> (TF_IDF ** PL2) 

# Learning models and re-ranking

fbr = pt.FeaturesBatchRetrieve(index, controls = {"wmodel": "BM25"}, features=["SAMPLE", "WMODEL:TF_IDF", "WMODEL:PL2"]) 

train_queries_path = os.path.join(DATA_FOLDER, LANGUAGES_FOLDER, language, BM25_FOLDER, "train", "queries.csv")
train_queries_df = pd.read_csv(train_queries_path)

#TODO: add a way to get training topics from data

# Uses pt.ltr.apply_learned_model(), because this is the new way to do this
# See: https://github.com/terrier-org/pyterrier/pull/432
BaselineLTR = fbr >> pt.ltr.apply_learned_model(RandomForestRegressor(n_estimators=400))
BaselineLTR.fit(train_queries_df, qrels)

results = pt.pipelines.Experiment([PL2, BaselineLTR], test_topics, qrels, ["map"], names=["PL2 Baseline", "LTR Baseline"])
results
