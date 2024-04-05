from collections import defaultdict
from typing import Dict, Tuple
import pyterrier as pt
import pandas as pd
from tqdm import tqdm
import datasets
import os
import json
import mlx.core as mlx
import mlx.nn as nn
from operator import itemgetter

languages_lib = [
    ['ar', 'arabic'],
    ['bn', 'bengali'],
    ['en', 'english'],
    ['es', 'spanish'],
    ['fa', 'persian'],
    ['fi', 'finnish'],
    ['fr', 'french'],
    ['hi', 'hindi'],
    ['id', 'indonesian'],
    ['ja', 'japanese'],
    ['ko', 'korean'],
    ['ru', 'russian'],
    ['sw', 'swahili'],
    ['te', 'telugu'],
    ['th', 'thai'],
    ['zh', 'chinese'],
    ['de', 'german'],
    ['yo', 'yoruba']
]

splits = [
    'train',
    'dev',
    'testA',
    'testB'
]

corpora = {}  # Store the MIRACL corpora for each language
queries = {}  # Store the MIRACL queries for each language


DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


def load_data_language(language: str) -> None:
    """
    Load the MIRACL dataset for a specific language.
    Refer to the MIRACL dataset for more information: https://huggingface.co/datasets/miracl/miracl

    Args:
        language (str): string representing the language of the dataset to load.
    """
    miracl_corpus = datasets.load_dataset(
        'miracl/miracl-corpus', language, trust_remote_code=True)
    miracl_queries = datasets.load_dataset(
        'miracl/miracl', language, trust_remote_code=True)
    corpora[language] = miracl_corpus
    queries[language] = miracl_queries
    print(f"Loaded MIRACL data for language {language}")


def load_all_miracle_data() -> None:
    """
    Load the MIRACL dataset for all languages.
    """
    languages = [lang[0] for lang in languages_lib]
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
    # Only a "train" split is available, = everything
    for doc in tqdm(corpora[language]["train"], desc=desc):
        yield {
            'id': doc['docid'],
            'doc_id': doc['docid'].split('#')[0],
            'passage_id': doc['docid'].split('#')[1],
            'title': doc['title'],
            'text': doc['text'],
        }


def filter_str(s: str) -> str:
    return s.replace("\n", "").replace("\t", "").replace("\r", "").replace("?", "").replace("'", "").replace("\"", "").replace(":", "").replace(";", "").replace("!", "").replace(".", "").replace(",", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("{", "").replace("}", "").replace("<", "").replace(">", "").replace("/", "").replace("\\", "").replace("|", "").replace("@", "").replace("#", "").replace("$", "").replace("%", "").replace("^", "").replace("&", "").replace("*", "").replace("+", "").replace("=", "").replace("~", "").replace("`", "").lower().strip()


def get_lengths_per_document(language: str) -> Dict[str, int]:
    """
    Get the lengths of the titles and texts of the documents in the corpus of a specific language.

    Args:
        language (str): string representing the language of the dataset to load.

    Returns:
        Dict[str, int]: a dictionary with the lengths of the doc_id and texts of the documents in the corpus.
    """
    assert language in corpora, f"Language {language} not loaded"

    lengths = defaultdict(int)
    for instance in tqdm(get_corpus_iter(language), desc=f"Getting lengths for {language}"):
        lengths['doc_id'] += len(instance['text'])
    print(f"Number of documents in {language} corpus: {len(lengths)}")
    return lengths


def get_unique_documents(language: str, top_freqs: int = 50) -> None:
    passages = set()
    documents = set()
    doc_lengths = defaultdict(int)
    title_lengths = []

    unigram_freq_doc = defaultdict(int)
    unigram_freq_title = defaultdict(int)

    digram_freq_doc = defaultdict(int)
    digram_freq_title = defaultdict(int)

    for instance in tqdm(get_corpus_iter(language), desc=f"Getting stats for {language}"):
        passages.add(instance['id'])
        documents.add(instance['doc_id'])
        doc_lengths[instance['doc_id']] += len(instance['text'])
        title_lengths.append(len(instance['title']))
        unigram_freq_doc[filter_str(
            " ".join(instance['text'].split()[:1]))] += 1
        unigram_freq_title[filter_str(
            " ".join(instance['title'].split()[:1]))] += 1
        digram_freq_doc[filter_str(
            " ".join(instance['text'].split()[:2]))] += 1
        digram_freq_title[filter_str(
            " ".join(instance['title'].split()[:2]))] += 1

    all_doc_lengths = mlx.array(list(doc_lengths.values()))
    all_title_lengths = mlx.array(title_lengths)

    title_avg = 0
    for title_idx in range(len(title_lengths)):
        title_avg = (title_idx / (title_idx + 1)) * title_avg + \
            (1 / (title_idx + 1)) * title_lengths[title_idx]

    doc_avg = 0
    for doc_idx, (doc_id, doc_length) in enumerate(doc_lengths.items()):
        doc_avg = (doc_idx / (doc_idx + 1)) * doc_avg + \
            (1 / (doc_idx + 1)) * doc_length

    stats = {
        "language": language,
        "passages": len(passages),
        "documents": len(documents),
        "mean_document_length": all_doc_lengths.mean().item(),
        "mean_title_length": all_title_lengths.mean().item(),
        "avg_document_length": doc_avg,
        "avg_title_length": title_avg,
        "min_document_length": all_doc_lengths.min().item(),
        "min_title_length": all_title_lengths.min().item(),
        "max_document_length": all_doc_lengths.max().item(),
        "max_title_length": all_title_lengths.max().item(),
        "unigram_freq_doc": dict(sorted(unigram_freq_doc.items(), key=itemgetter(1), reverse=True)[:top_freqs]),
        "unigram_freq_title": dict(sorted(unigram_freq_title.items(), key=itemgetter(1), reverse=True)[:top_freqs]),
        "digram_freq_doc": dict(sorted(digram_freq_doc.items(), key=itemgetter(1), reverse=True)[:top_freqs]),
        "digram_freq_title": dict(sorted(digram_freq_title.items(), key=itemgetter(1), reverse=True)[:top_freqs])
    }

    if not os.path.exists(os.path.join(DATA_FOLDER, "analysis", "corpora")):
        os.makedirs(os.path.join(DATA_FOLDER, "analysis", "corpora"))
    with open(os.path.join(DATA_FOLDER, "analysis", "corpora", f"{language}_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Succesfully saved stats for {language} corpus")
    return


def get_query_stats(language: str, top_freqs: int = 50):

    stats = {
        "language": language,
    }

    total_unigram_freq = defaultdict(int)
    total_digram_freq = defaultdict(int)
    total_query_lengths = []
    total_avg = 0

    for split in splits:
        if split not in queries[language]:
            continue
        split_queries = queries[language][split]
        split_unigram_freq = defaultdict(int)
        split_digram_freq = defaultdict(int)
        query_lengths = []
        avg = 0
        for query in split_queries:
            query_lengths.append(len(query['query']))
            avg = (len(query_lengths) / (len(query_lengths) + 1)) * \
                avg + (1 / (len(query_lengths) + 1)) * len(query['query'])
            split_unigram_freq[filter_str(
                " ".join(query['query'].split()[:1]))] += 1
            split_digram_freq[filter_str(
                " ".join(query['query'].split()[:2]))] += 1

        total_query_lengths.extend(query_lengths)
        for key in split_unigram_freq:
            total_unigram_freq[key] += split_unigram_freq[key]
        for key in split_digram_freq:
            total_digram_freq[key] += split_digram_freq[key]

        all_query_lengths = mlx.array(query_lengths)
        split_stats = {
            "split": split,
            "queries": len(split_queries),
            "mean_query_length": all_query_lengths.mean().item(),
            "avg_query_length": avg,
            "min_query_length": all_query_lengths.min().item(),
            "max_query_length": all_query_lengths.max().item(),
            "unigram_freq": dict(sorted(split_unigram_freq.items(), key=itemgetter(1), reverse=True)[:top_freqs]),
            "digram_freq": dict(sorted(split_digram_freq.items(), key=itemgetter(1), reverse=True)[:top_freqs])
        }
        stats[split] = split_stats
        total_avg = (len(total_query_lengths) / (len(total_query_lengths) + len(query_lengths))) * \
            total_avg + (len(query_lengths) /
                         (len(total_query_lengths) + len(query_lengths))) * avg

    all_total_query_lengths = mlx.array(total_query_lengths)
    stats["total"] = {
        "split": "total",
        "queries": len(total_query_lengths),
        "mean_query_length": all_total_query_lengths.mean().item(),
        "avg_query_length": total_avg,
        "min_query_length": all_total_query_lengths.min().item(),
        "max_query_length": all_total_query_lengths.max().item(),
        "unigram_freq": dict(sorted(total_unigram_freq.items(), key=itemgetter(1), reverse=True)[:top_freqs]),
        "digram_freq": dict(sorted(total_digram_freq.items(), key=itemgetter(1), reverse=True)[:top_freqs])
    }

    if not os.path.exists(os.path.join(DATA_FOLDER, "analysis", "queries")):
        os.makedirs(os.path.join(DATA_FOLDER, "analysis", "queries"))
    with open(os.path.join(DATA_FOLDER, "analysis", "queries", f"{language}_stats.json"), 'w') as f:
        json.dump(stats, f, indent=4)

    print(f"Succesfully saved stats for {language} queries")
    return


def get_data_stats(language: str):
    print()
    get_unique_documents(language)
    get_query_stats(language)
    print()
    print(f"Stats for {language} corpus and queries have been saved")


if __name__ == '__main__':
    load_all_miracle_data()
    languages = [lang[0] for lang in languages_lib]
    # concurrently get_data_stats for all languages
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor() as executor:
        list(executor.map(get_data_stats, languages))
