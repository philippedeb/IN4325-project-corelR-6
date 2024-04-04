import pyterrier as pt
import datasets
import pandas as pd
from tqdm import tqdm
import os
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords

# Initialize required nltk downloads
nltk.download('stopwords')

LANG_DICT = {       # languages supported by snowball stemmer
    'ar': 'arabic',
    'de': 'german',
    'en': 'english',
    'es': 'spanish',
    'fi': 'finnish',
    'fr': 'french',
    'ru': 'russian'
}

def init_pyterrier():
    if not pt.started():
        pt.init(boot_packages=[
            "io.anserini:anserini:0.25.0:fatjar",
            "com.github.terrierteam:terrier-prf:-SNAPSHOT"])

class DataLoader:
    def __init__(self, language_code):
        self.language_code = language_code
        self.language = LANG_DICT[language_code]
        self.corpus_path = f'../MIRACL-corpora/miracl-corpus-huggingface-{language_code}'
        self.queries_path = f'../MIRACL-queries/miracl-queries-huggingface-{language_code}'
        self.corpus, self.queries = self.load_data()

    def load_or_save_dataset(self, name, save_path):
        if not os.path.exists(save_path):
            print(f"Dataset does not exists yet. Downloading {name} in {self.language}...")
            dataset = datasets.load_dataset(name, self.language_code, trust_remote_code=True)
            dataset.save_to_disk(save_path)
        else:
            print(f"Dataset already exists. Loading dataset from {save_path}...")
            dataset = datasets.load_from_disk(save_path)
        return dataset

    def load_data(self):
        corpus = self.load_or_save_dataset('miracl/miracl-corpus', self.corpus_path)
        queries = self.load_or_save_dataset('miracl/miracl', self.queries_path)
        print(f"Loaded MIRACL data for language {self.language}")
        return corpus, queries

class TextPreprocessor:
    def __init__(self, language_code):
        self.language = LANG_DICT[language_code]
        self.stemmer = SnowballStemmer(self.language)
        self.stop_words = set(stopwords.words(self.language))

    def preprocess_text(self, text, is_query=False):
        text = text.lower()
        text = re.sub(r'\W+', ' ', text)
        words = text.split()
        if is_query:
            filtered_words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        else:
            filtered_words = [self.stemmer.stem(word) for word in words]  # Stem all words in documents
        
        return ' '.join(filtered_words)

def prepare_queries(language, split, preprocessor=None, preprocess=False):
    queries_file_path = f'tools/topics-and-qrels/topics.miracl-v1.0-{language}-{split}.tsv'
    if not os.path.exists(queries_file_path):
        raise FileNotFoundError(f"Query file not found: {queries_file_path}")
    
    queries_df = pd.read_csv(queries_file_path, sep='\t', header=None, names=['qid', 'query'])

    # Convert 'qid' to string to match between files
    queries_df['qid'] = queries_df['qid'].astype(str)
    
    if preprocess and preprocessor:
        queries_df['query'] = queries_df['query'].apply(lambda x: preprocessor.preprocess_text(x, is_query=True))
    else:
        queries_df['query'] = queries_df['query'].apply(lambda x: re.sub(r'\W+', ' ', x))

    return queries_df

def prepare_qrels(lang, split):
    qrels_file_path = 'tools/topics-and-qrels/qrels.miracl-v1.0-' + lang + '-' + split + '.tsv'

    # Load qrels
    qrels_df = pd.read_csv(qrels_file_path, sep='\t', header=None, usecols=[0, 2, 3], names=['qid', 'docno', 'label'])

    # Convert 'qid' to string to match between files
    qrels_df['qid'] = qrels_df['qid'].astype(str)

    return qrels_df

def miracl_corpus_iter(corpus):
    for doc in corpus['train']:
        yield {
            'docno': doc['docid'],
            'title': doc['title'],
            'text': doc['text']
        }

def miracl_corpus_iter_preprocessed(corpus, language, preprocessor):
    """Corpus iterator with language-specific preprocessing."""
    for doc in corpus['train']:
        yield {
            'docno': doc['docid'],
            'title': preprocessor.preprocess_text(doc['title'], is_query=False),
            'text': preprocessor.preprocess_text(doc['text'], is_query=False)
        }

def get_index_retriever(language, index_type, preprocessor=None):
    if index_type == 'prebuilt':    # special handling for prebuilt pyserini index
        index_path = f'../MIRACL-indexes/lucene-index.miracl-v1.0-{language}'
        return pt.anserini.AnseriniBatchRetrieve(index_path, wmodel="BM25")
    else:   # custom pyterrier index construction
        index_dir = f"../MIRACL-indexes/miracl-index-{language}-{index_type}"
        if not os.path.exists(index_dir) or not os.listdir(index_dir):
            print(f"Index of language '{language}' does not exist yet. Building new index in {index_dir}...")
            indexer = pt.IterDictIndexer(index_dir, overwrite=True, blocks=True)
            if index_type == 'preprocessed':
                corpus_iter = miracl_corpus_iter_preprocessed(DataLoader(language).corpus, language, preprocessor)
            else:  # raw
                corpus_iter = miracl_corpus_iter(DataLoader(language).corpus)
            indexer.index(corpus_iter)
        print(f"Loading index from {index_dir}...")
        index_ref = pt.IndexFactory.of(index_dir)
        return pt.BatchRetrieve(index_ref, wmodel="BM25")

def run_experiment(queries_df, qrels_df, retriever, output_path, experiment_name):
    results = pt.Experiment(
        [retriever],
        queries_df,
        qrels_df,
        ["ndcg_cut_10", "recall_100"],
        names=[experiment_name]
    )
    results.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
    print(f"Results for {experiment_name} saved to {output_path}")

def main(language):
    print("LANGUAGE: ", language)
    init_pyterrier()
    preprocessor = TextPreprocessor(language)

    split = 'dev'
    
    qrels_df = prepare_qrels(language, split)  

    # ### Explanation Experiment Names - different combination of queries and indices
    # Query datframe types
    # Q_pp ... queries preprocessed
    # Q_raw ... queries raw (only with punctuation remover)
    # ###
    # Index types
    # prebuilt ... prebuilt lucene index from pyserini
    # custom_pp ... custom index created with pyterrier, with preprocessing
    # custom_raw ... custom index created with pyterrier, without preprocessing
    # ###

    experiments = [
        {"preprocess_queries": True, "index_type": "prebuilt", "experiment_name": "Q_pp-I_prebuilt"},
        {"preprocess_queries": True, "index_type": "preprocessed", "experiment_name": "Q_pp-I_custom_pp"},
        {"preprocess_queries": True, "index_type": "raw", "experiment_name": "Q_pp-I_custom_raw"},
        {"preprocess_queries": False, "index_type": "prebuilt", "experiment_name": "Q_raw-I_prebuilt"},
        {"preprocess_queries": False, "index_type": "preprocessed", "experiment_name": "Q_raw-i_custom_pp"},
        {"preprocess_queries": False, "index_type": "raw", "experiment_name": "Q_raw-I_custom_raw"}
    ]

    for i, exp in enumerate(experiments):
        experiment_name = exp["experiment_name"]
        print(f"Experiment {i+1}/{len(experiments)}: {experiment_name}")

        print("Prepare Queries")
        queries_df = prepare_queries(language, split, preprocessor, exp["preprocess_queries"])
        print("Prepare Index")
        retriever = get_index_retriever(language, exp["index_type"], preprocessor if exp["index_type"] == 'preprocessed' else None)
        print("Run Experiment")
        run_experiment(queries_df, qrels_df, retriever, f"../results/bm25_results_{language}.csv", exp["experiment_name"])

if __name__ == '__main__':
    main('fi')