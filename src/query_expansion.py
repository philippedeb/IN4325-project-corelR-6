import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import datasets
import os
import json
import time
from typing import Tuple
from unidecode import unidecode
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import string
# import numpy as np
import random

nltk.download('punkt')

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

if not pt.started():
    # pt.init(boot_packages=["com.github.terrierteam:terrier-prf:-SNAPSHOT"])
    pt.init()
    
corpora = {}
queries = {}
processed_docs = []
stop_words = {
    'fr': set(stopwords.words('french')),
    'fi': set(stopwords.words('finnish')),
    'de': set(stopwords.words('german')),
    'en': set(stopwords.words('english')),
    'es': set(stopwords.words('spanish')),
}
    
def load_data_language(language: str) -> None:
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
    
    tokenize_docs(language)
    # print("Without expansion")
    
    for idx, data in enumerate(tqdm(queries[language][split], desc="Processing Queries and Qrels")):
        expanded_query = word2vec(data['query'], language)
        # queries_pyt.append({'qid': data['query_id'], 'query': unidecode(expanded_query)})
        queries_pyt.append({'qid': data['query_id'], 'query': expanded_query})
        # queries_pyt.append({'qid': data['query_id'], 'query': unidecode(data['query'])})
        # queries_pyt.append({'qid': data['query_id'], 'query': data['query']})
        # queries_pyt.append({'qid': data['query_id'], 'query': expand_query(unidecode(data['query']))})
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
    queries_df['query'] = queries_df['query'].str.replace("¡", "")  # 
    queries_df['query'] = queries_df['query'].str.replace("¿", "")  # 
    # queries_df['query'] = queries_df['query'].str.replace(".", "")  # 
    # queries_df['query'] = queries_df['query'].str.replace(",", "")  # 
    # queries_df['query'] = queries_df['query'].str.replace("^", "")  # 

    qrels_df = pd.DataFrame(qrels_pyt)

    return queries_df, qrels_df

def preprocess(text):
    return text.translate(str.maketrans('', '', string.punctuation))

def tokenize_docs(language):
    fraction = 0.0001
    corpora_length = len(corpora[language]["train"])
    print(f"Tokenizing {int(corpora_length*fraction)} documents")
    rand_ind = [random.randint(0, corpora_length - 1) for _ in range(int(corpora_length * fraction))]
    for i in rand_ind:
    # for i in range(int(corpora_length * fraction)):
        processed_docs.extend([word_tokenize(preprocess(corpora[language]["train"][i]["text"]))])

def word2vec(query, language):
    tokenized_queries = word_tokenize(preprocess(query))
    all_sentences = tokenized_queries + processed_docs
    model = Word2Vec(sentences=all_sentences, vector_size=300, window=10, min_count=10, workers=4)
    exp_query = list(tokenized_queries)
    stop_word = stop_words.get(language, set())

    for word in tokenized_queries:
        if word in stop_word:
            continue
        
        if word in model.wv:
            similar_words = model.wv.most_similar(word)
            for key, value in similar_words:
                if key in stop_word:
                    continue
                if value >= 0.8:
                    exp_query.append(key)
                    break
    
    return ' '.join(exp_query)

def index_miracl_corpus(language: str):
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

    if language == "de":
        print("German indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True, stemmer="GermanSnowballStemmer", stopwords=None, tokeniser="UTFTokeniser")
        return indexer.index(get_corpus_iter(language))
    elif language == "fr":
        print("French indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True, stemmer="FrenchSnowballStemmer", stopwords=None, tokeniser="UTFTokeniser")
        return indexer.index(get_corpus_iter(language))
    elif language == "fi":
        print("Finnish indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True, stemmer="FinnishSnowballStemmer", stopwords=None, tokeniser="UTFTokeniser")
        return indexer.index(get_corpus_iter(language))
    elif language == "es":
        print("Spanish indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True, stemmer="SpanishSnowballStemmer", stopwords=None, tokeniser="UTFTokeniser")
        return indexer.index(get_corpus_iter(language))
    elif language == "en":
        print("English indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True)
        return indexer.index(get_corpus_iter(language))
    else:
        print("Standard indexer")
        indexer = pt.IterDictIndexer(miracl_index_path, overwrite=True, blocks=True, stopwords=None, tokeniser="UTFTokeniser")
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

    print(f"Loaded index for {language}")

    # Load the queries and qrels
    # queries_path = os.path.join(BM25_SPLIT_FOLDER, 'queries.csv')
    # qrels_path = os.path.join(BM25_SPLIT_FOLDER, 'qrels.csv')

    # If queries and qrels do not exist, create them
    # if not os.path.exists(queries_path) or not os.path.exists(qrels_path):
    #     queries_df, qrels_df = get_dataframe_queries_qrels(language, split)
    #     queries_df.to_csv(queries_path, index=False)
    #     qrels_df.to_csv(qrels_path, index=False)
    # else:
        # queries_df = pd.read_csv(queries_path)
        # qrels_df = pd.read_csv(qrels_path)

    queries_df, qrels_df = get_dataframe_queries_qrels(language, split)
    print(f"Loaded queries and qrels for {language} ({split})")
    
    # Define the BM25 retrieval model
    # bo1 = pt.rewrite.Bo1QueryExpansion(index)
    # kl = pt.rewrite.KLQueryExpansion(index)
    # rm3 = pt.rewrite.RM3(index)
    # sd =  pt.rewrite.SequentialDependence(index)
    # axiomatic = pt.rewrite.AxiomaticQE(index)
    
    # tfidf_vectorizer = TfidfVectorizer()
    # lsa_model = TruncatedSVD(n_components=10)
    BM25 = pt.BatchRetrieve(index, wmodel="BM25")
    # qe_bo1 = BM25 >> bo1 >> BM25
    # qe_kl = BM25 >> kl >> BM25
    # qe_rm3 = BM25 >> rm3 >> BM25
    # qe_sd = BM25 >> sd >> BM25
    # qe_ax = BM25 >> axiomatic >> BM25

    # Retrieve the results
    # results = BM25.transform(queries_df)

    eval_metrics = ['map', 'ndcg']
    eval_results = pt.Experiment(
        [BM25],
        queries_df,
        qrels_df,
        eval_metrics=eval_metrics,
        names=["BM25"]
    )

    print(f"Evaluated BM25 for {language} ({split})")
    print(eval_results)

    # Write the results to a file
    eval_results.to_csv(os.path.join(BM25_SPLIT_FOLDER, time.strftime("%Y%m%d-%H%M%S") + '_results.csv'))

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
    
    language = "es"
    load_data_language(language)
    # get_bm25_results(language, "dev")
