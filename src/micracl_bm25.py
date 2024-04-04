from typing import Tuple
import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import datasets
import os
import json

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

# Initialize PyTerrier
if not pt.started():
    pt.init()

corpora = {}  # Store the MIRACL corpora for each language
queries = {}  # Store the MIRACL queries for each language


def load_data_language(language: str) -> None:
    """
    Load the MIRACL dataset for a specific language.
    Refer to the MIRACL dataset for more information: https://huggingface.co/datasets/miracl/miracl

    Args:
        language (str): string representing the language of the dataset to load.
    """
    miracl_corpus = datasets.load_dataset('miracl/miracl-corpus', language, trust_remote_code=True)
    miracl_queries = datasets.load_dataset('miracl/miracl', language, trust_remote_code=True)
    corpora[language] = miracl_corpus
    queries[language] = miracl_queries
    print(f"Loaded MIRACL data for language {language}")


def load_all_miracle_data() -> None:
    """
    Load the MIRACL dataset for all languages.
    """
    languages = json.load(open(os.path.join(DATA_FOLDER, 'languages.json'))).keys()
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # Initially 2h1m54s, around 13s when preloaded
        list(executor.map(load_data_language, languages))


def get_corpus_iter(language: str):
    """
    Get an iterator over the corpus of a specific language.

    Args:
        language (str): string representing the language of the dataset to load.

    Returns:
        Iterator: an iterator over the corpus of a specific language.
    """
    assert language in corpora, f"Language {language} not loaded"

    desc = f"Processing {language} corpus"
    for doc in tqdm(corpora[language]["train"], desc=desc):  # Only a "train" split is available, = everything
        yield {
            'docno': doc['docid'],
            'title': doc['title'],
            'text': doc['text']
        }


def get_dataframe_queries_qrels(language: str, split: str = "dev") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Get a dataframe containing the queries and qrels of a specific language.

    Args:
        language (str): string representing the language of the dataset to load.
        split (str): string representing the split of the dataset to load. Default is "dev".

    Returns:
        DataFrame: a dataframe containing the queries and qrels of a specific language.
    """
    assert language in queries, f"Language {language} not loaded"
    assert split in queries[language], f"Split {split} not found for language {language}"

    # Preparing queries and qrels for PyTerrier
    queries_pyt = []
    qrels_pyt = []

    for idx, data in enumerate(tqdm(queries[language][split], desc="Processing Queries and Qrels")):
        queries_pyt.append({'qid': data['query_id'], 'query': data['query']})
        for entry in data['positive_passages']:
            qrels_pyt.append({'qid': data['query_id'],
                        'docno': entry['docid'], 'label': 1})
        for entry in data['negative_passages']:
            qrels_pyt.append({'qid': data['query_id'],
                        'docno': entry['docid'], 'label': 0})


    queries_df = pd.DataFrame(queries_pyt)
    queries_df['query'] = queries_df['query'].str.replace('?', '')  # remove question marks
    queries_df['query'] = queries_df['query'].str.replace("'", "")  # remove apostrophes
    queries_df['query'] = queries_df['query'].str.replace("/", "")  # remove slash
    queries_df['query'] = queries_df['query'].str.replace(":", "")  # remove colon
    queries_df['query'] = queries_df['query'].str.replace("!", "")  # remove exclamation mark


    qrels_df = pd.DataFrame(qrels_pyt)

    return queries_df, qrels_df


def index_miracl_corpus(language: str):
    """
    Indexes the MIRACL corpus for a specified language.

    Args:
        language (str): The language for which the MIRACL corpus will be indexed.

    Returns:
        Indexed data obtained by indexing the MIRACL corpus for the specified language.
    """
    assert language in corpora, f"Language {language} not loaded"

    # Set up the folder to store the indexed files
    LANGUAGES_FOLDER = os.path.join(DATA_FOLDER, 'languages')
    if not os.path.exists(LANGUAGES_FOLDER):
        os.makedirs(LANGUAGES_FOLDER)
    LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, language)
    if not os.path.exists(LANGUAGE_FOLDER):
        os.makedirs(LANGUAGE_FOLDER)
    INDICES_FOLDER = os.path.join(LANGUAGE_FOLDER, 'indices')
    if not os.path.exists(INDICES_FOLDER):
        os.makedirs(INDICES_FOLDER)
    SPLIT_FOLDER = os.path.join(INDICES_FOLDER, "train")  # Only a "train" split is available, = everything
    if not os.path.exists(SPLIT_FOLDER):
        os.makedirs(SPLIT_FOLDER)
    miracl_index_path = os.path.join(SPLIT_FOLDER, 'miracl_index')

    indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True)
    return indexer.index(get_corpus_iter(language))


def get_bm25_results(language: str, split: str = "dev"):
    assert language in corpora, f"Language {language} not loaded"
    assert split in queries[language], f"Split {split} not found for language {language}"

    # Set up the folder to store the results
    LANGUAGES_FOLDER = os.path.join(DATA_FOLDER, 'languages')
    LANGUAGE_FOLDER = os.path.join(LANGUAGES_FOLDER, language)

    BM25_FOLDER = os.path.join(LANGUAGE_FOLDER, 'bm25')
    if not os.path.exists(BM25_FOLDER):
        os.makedirs(BM25_FOLDER)
    
    BM25_SPLIT_FOLDER = os.path.join(BM25_FOLDER, split)
    if not os.path.exists(BM25_SPLIT_FOLDER):
        os.makedirs(BM25_SPLIT_FOLDER)

    INDICES_FOLDER = os.path.join(LANGUAGE_FOLDER, 'indices')
    INDEX_SPLIT_FOLDER = os.path.join(INDICES_FOLDER, "train")
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


def get_bm25_results(language: str, split: str = "dev"):
    setup_bm25_folders(language, split)

    # Load the index
    index = pt.IndexFactory.of(miracl_index_path)

    print(f"Loaded index for {language}")

    # Load the queries and qrels
    queries_path = os.path.join(BM25_SPLIT_FOLDER, 'queries.csv')
    qrels_path = os.path.join(BM25_SPLIT_FOLDER, 'qrels.csv')

    # If queries and qrels do not exist, create them
    if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
        queries_df, qrels_df = get_dataframe_queries_qrels(language, split)
        queries_df.to_csv(queries_path, index=False)
        qrels_df.to_csv(qrels_path, index=False)
    else:
        queries_df = pd.read_csv(queries_path)
        qrels_df = pd.read_csv(qrels_path)

    print(f"Loaded queries and qrels for {language} ({split})")

    # Define the BM25 retrieval model
    BM25 = pt.BatchRetrieve(index, wmodel="BM25")

    # Retrieve the results
    results = BM25.transform(queries_df)

    eval_metrics = ['map', 'ndcg']
    eval_results = pt.Experiment(
        [results],
        queries_df,
        qrels_df,
        eval_metrics=eval_metrics
    )

    print(f"Evaluated BM25 for {language} ({split})")
    print(eval_results)

    # Write the results to a file
    eval_results.to_csv(os.path.join(BM25_SPLIT_FOLDER, 'results.csv'))

    return eval_results


def get_all_bm25_results():
    # for language in corpora.keys():
    #     for split in corpora[language].keys():
    #         print(f"\n>>> Processing {language} ({split})\n")
    #         get_bm25_results(language, split)

    # Do the code above for all languages and splits, but in a concurrent way,
    # using the ThreadPoolExecutor from the concurrent.futures module.
    # Furthermore, use a try/catch block to handle exceptions and store them in a list.
    # At the end, print out which languages/splits failed. Not the exception itself, just the language/split.
    # The try/catch block is there such that the code does not stop running.
    
    import concurrent.futures
    failed = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for language in corpora.keys():
            for split in queries[language].keys():
                try:
                    print(f"\n>>> Processing {language} ({split})\n")
                    executor.submit(get_bm25_results, language, split)
                except Exception as e:
                    failed.append(f"{language} ({split})")


if __name__ == '__main__':

    # Get all the results for BM25
    # load_all_miracle_data()
    # get_all_bm25_results()
    
    # Get BM25 results for a specific language and split
    load_data_language("en")
    get_bm25_results("en", "testA")
