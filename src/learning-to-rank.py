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
    Get a dataframe containing the queries and qrels of a specific language.

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

lang='sw' # choose language
load_data_language(lang)
qrels = get_dataframe_queries_qrels(lang)

print(qrels)


# if __name__ == '__main__':
    # Get BM25 results for a specific language and split
  #  load_data_language("sw")
  #  get_bm25_results("sw", "testA")
