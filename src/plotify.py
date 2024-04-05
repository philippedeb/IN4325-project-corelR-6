from collections import defaultdict
from typing import Dict, Tuple
from matplotlib import cm
import pandas as pd
from tqdm import tqdm
import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.font_manager import FontProperties
import yaml

prop = FontProperties()

# font file apple unicode


DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
font = os.path.join(DATA_FOLDER, 'GoNotoCJKCore.ttf')
prop.set_file(font)
prop.set_file('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')

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

query_splits_stats = [
    'train',
    'dev',
    'testA',
    'testB',
    'total'
]

fontsize = {
    "digram": [
        ['ar', 30],
        ['bn', 30],
        ['en', 30],
        ['es', 30],
        ['fa', 30],
        ['fi', 30],
        ['fr', 30],
        ['hi', 30],
        ['id', 27],
        ['ja', 10],
        ['ko', 27],
        ['ru', 30],
        ['sw', 30],
        ['te', 23],
        ['th', 23],
        ['zh', 13],
        ['de', 30],
        ['yo', 30]
    ],
    "unigram": [
        ['ar', 30],
        ['bn', 30],
        ['en', 30],
        ['es', 30],
        ['fa', 30],
        ['fi', 30],
        ['fr', 30],
        ['hi', 30],
        ['id', 30],
        ['ja', 30],
        ['ko', 30],
        ['ru', 30],
        ['sw', 30],
        ['te', 30],
        ['th', 27],
        ['zh', 13],
        ['de', 30],
        ['yo', 30]
    ]
}

DATA_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
STATS_FOLDER = os.path.join(DATA_FOLDER, 'analysis')
CORPORA_FOLDER = os.path.join(STATS_FOLDER, 'corpora')
QUERIES_FOLDER = os.path.join(STATS_FOLDER, 'queries')


def get_query_frequencies(language: str, stats_split: str = "total", diagram: str = "unigram"):
    query_data = json.load(
        open(os.path.join(QUERIES_FOLDER, f'{language}_stats.json')))
    stats = query_data[stats_split]
    total = stats["queries"]
    freqs = stats[f"{diagram}_freq"]
    return freqs, total


def get_corpora_frequency(language: str, diagram: str = "unigram", data: str = "doc"):
    corpora_data = json.load(
        open(os.path.join(CORPORA_FOLDER, f'{language}_stats.json')))
    total = corpora_data["passages"]
    freqs = corpora_data[f"{diagram}_freq_{data}"]
    return freqs, total


def query_polar_barplot(language: str, total: int, frequencies: Dict[str, int], k: int = 10, title: bool = False, color_bar: bool = False, show: bool = False, fontsize: int = 16):

    freq_data = frequencies
    # Get only the first k of unigram_freq
    freq_data = dict(list(freq_data.items())[:k])

    # Convert frequencies to percentages
    freq_data = {word: count / total *
                 100 for word, count in freq_data.items()}

    # Get words and their corresponding counts sorted by count
    long_words = any([len(word) > 15 for word in freq_data.keys()])
    words = [f"{word}\n({(perc):.2f}%)" if len(word) <= 15 else f"{word}\n({(perc):.2f}%)".replace(
        " ", "\n") for word, perc in freq_data.items()]
    counts = list(freq_data.values())

    # Calculate angles for polar plot
    angles = np.linspace(0, 2 * np.pi, len(words), endpoint=False).tolist()

    # Define a colormap from blue to red
    from matplotlib import colormaps
    colormap = colormaps.get_cmap('cool')

    # Normalize the counts for colors
    norm = plt.Normalize(min(counts), max(counts))

    # Plot the polar plot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})

    # Set size of the plot
    fig.set_size_inches(16, 10)

    # Change grid to be circular
    ax.grid(True)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    # Gridlines behind the bars
    ax.set_axisbelow(True)

    bars = ax.bar(angles, counts, width=0.5,
                  color=colormap(norm(counts)), edgecolor=colormap(norm(counts)))

    # Rotate the labels to avoid clashing
    rotation = np.degrees(angles)
    rotation = list(reversed(rotation))
    rotation = [rotation[-1]] + rotation[:-1]
    rotation = [r + 180 if r > 90 and r < 270 else r for r in rotation]

    # Set the labels for each bar
    ax.set_xticks(angles)

    # For each word in words, set the corresponding label with the corresponding rotation
    # ax.set_xticklabels(words,
    #                    fontsize=16, fontproperties=prop)

    labels = []
    for i, (label, angle) in enumerate(zip(ax.get_xticklabels(), rotation)):
        x, y = label.get_position()
        lab = ax.text(x, y, words[i], transform=label.get_transform(),
                      ha=label.get_ha(), va=label.get_va(), fontproperties=prop, fontsize=fontsize)
        lab.set_rotation(angle)
        labels.append(lab)
    ax.set_xticklabels([])

    # Pad the labels
    ax.tick_params(axis='x', pad=fontsize + fontsize //
                   2 if long_words else fontsize)

    # Add color bar
    if color_bar:
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Frequency')

    # Pad the title and set size
    if title:
        plt.title(f'Top {k} starting unigrams for {[y for x, y in languages_lib if x == language][0].capitalize()} ',
                  pad=20, fontsize=20)

    # Remove y labels
    ax.set_yticklabels([])

    plt.tight_layout()
    if show:
        plt.show()


def get_all_query_frequency_plots():
    QUERY_PLOTS = os.path.join(QUERIES_FOLDER, 'plots')
    DIGRAMS = os.path.join(QUERY_PLOTS, 'digrams')
    UNIGRAMS = os.path.join(QUERY_PLOTS, 'unigrams')

    if not os.path.exists(QUERY_PLOTS):
        os.makedirs(QUERY_PLOTS)
    if not os.path.exists(DIGRAMS):
        os.makedirs(DIGRAMS)
    if not os.path.exists(UNIGRAMS):
        os.makedirs(UNIGRAMS)

    for lang_short, lang_full in languages_lib:
        for split in query_splits_stats:
            data = json.load(
                open(os.path.join(QUERIES_FOLDER, f'{lang_short}_stats.json')))
            if split not in data:
                continue
            for diagram in ["unigram", "digram"]:
                filename = f"{lang_short}_{split}_{diagram}.png"
                frequencies, total = get_query_frequencies(
                    lang_short, split, diagram=diagram)
                query_polar_barplot(lang_short, total, frequencies, k=10, fontsize=[
                                    y for x, y in fontsize[diagram] if x == lang_short][0])
                plt.savefig(os.path.join(os.path.join(
                    QUERY_PLOTS, f"{diagram}s"), filename), bbox_inches='tight')
                plt.close()
        print(f"-- {lang_full}: query plots have been saved.")

    print("All plots have been saved.")
    return


def get_all_doc_frequency_plots():
    DOC_PLOTS = os.path.join(CORPORA_FOLDER, 'plots')
    DOC_FOLDER = os.path.join(DOC_PLOTS, 'docs')
    DOC_DIAGRAMS = os.path.join(DOC_FOLDER, 'digrams')
    DOC_UNIGRAMS = os.path.join(DOC_FOLDER, 'unigrams')
    TITLE_FOLDER = os.path.join(DOC_PLOTS, 'titles')
    TITLE_DIAGRAMS = os.path.join(TITLE_FOLDER, 'digrams')
    TITLE_UNIGRAMS = os.path.join(TITLE_FOLDER, 'unigrams')

    if not os.path.exists(DOC_PLOTS):
        os.makedirs(DOC_PLOTS)
    if not os.path.exists(DOC_FOLDER):
        os.makedirs(DOC_FOLDER)
    if not os.path.exists(DOC_DIAGRAMS):
        os.makedirs(DOC_DIAGRAMS)
    if not os.path.exists(DOC_UNIGRAMS):
        os.makedirs(DOC_UNIGRAMS)
    if not os.path.exists(TITLE_FOLDER):
        os.makedirs(TITLE_FOLDER)
    if not os.path.exists(TITLE_DIAGRAMS):
        os.makedirs(TITLE_DIAGRAMS)
    if not os.path.exists(TITLE_UNIGRAMS):
        os.makedirs(TITLE_UNIGRAMS)

    for lang_short, lang_full in languages_lib:
        for data in ["doc", "title"]:
            for diagram in ["unigram", "digram"]:
                filename = f"{lang_short}_{diagram}.png"
                frequencies, total = get_corpora_frequency(
                    lang_short, diagram=diagram, data=data)
                query_polar_barplot(lang_short, total, frequencies, k=10, fontsize=[
                                    y for x, y in fontsize[diagram] if x == lang_short][0])
                plt.savefig(os.path.join(os.path.join(
                    DOC_PLOTS, f"{data}s", f"{diagram}s"), filename), bbox_inches='tight')
                plt.close()
            print(f"-- {lang_full}: {data} plots have been saved.")

    print("All plots have been saved.")
    return


def get_latex_subfigures_code_queries_digrams():
    QUERY_PLOTS = os.path.join(QUERIES_FOLDER, 'plots')
    DIGRAMS = os.path.join(QUERY_PLOTS, 'digrams')

    # Go through all filenames in the diagrams and unigrams directory
    for filename in os.listdir(DIGRAMS):
        name = filename
        language = name.split("_")[0]
        split = name.split("_")[1]
        if split != "total":
            continue
        full_language = [y for x, y in languages_lib if x == language][0]
        caption = f"{full_language.capitalize()} queries, top starting-digrams."
        label = f"plot:polar:queries:digrams:{language}-{split}"
        res = "\\begin{subfigure}[b]{0.3\\textwidth}\n\t\\centering\n\t\\includegraphics[width=\\textwidth]{images/plots/queries/digrams/" + \
            name + "}\n\t\\caption{\\mbox{" + caption + \
            "}}\n\t\\label{fig:" + label + "}\n\\end{subfigure}"
        print(res)


def get_latex_subfigures_code_queries_unigrams():
    QUERY_PLOTS = os.path.join(QUERIES_FOLDER, 'plots')
    UNIGRAMS = os.path.join(QUERY_PLOTS, 'unigrams')

    # Go through all filenames in the diagrams and unigrams directory
    for filename in os.listdir(UNIGRAMS):
        name = filename
        language = name.split("_")[0]
        split = name.split("_")[1]
        if split != "total":
            continue
        full_language = [y for x, y in languages_lib if x == language][0]
        caption = f"{full_language.capitalize()} queries, top starting-unigrams."
        label = f"plot:polar:queries:unigrams:{language}-{split}"
        res = "\\begin{subfigure}[b]{0.3\\textwidth}\n\t\\centering\n\t\\includegraphics[width=\\textwidth]{images/plots/queries/unigrams/" + \
            name + "}\n\t\\caption{\\mbox{" + caption + \
            "}}\n\t\\label{fig:" + label + "}\n\\end{subfigure}"
        print(res)


def get_latex_subfigures_code_docs_unigrams():
    DOC_PLOTS = os.path.join(CORPORA_FOLDER, 'plots')
    DOC_FOLDER = os.path.join(DOC_PLOTS, 'docs')
    DOC_UNIGRAMS = os.path.join(DOC_FOLDER, 'unigrams')

    # Go through all filenames in the diagrams and unigrams directory
    for filename in os.listdir(DOC_UNIGRAMS):
        name = filename
        language = name.split("_")[0]
        full_language = [y for x, y in languages_lib if x == language][0]
        caption = r"\small{" + \
            f"{full_language.capitalize()} passages, top starting-unigrams." + "}"
        label = f"plot:polar:passages:unigrams:{language}"
        res = "\\begin{subfigure}[b]{0.3\\textwidth}\n\t\\centering\n\t\\includegraphics[width=\\textwidth]{images/plots/documents/unigrams/" + \
            name + "}\n\t\\caption{\\footnotesize{" + caption + \
            "}}\n\t\\label{fig:" + label + "}\n\\end{subfigure}"
        print(res)


def get_latex_subfigures_code_docs_digrams():
    DOC_PLOTS = os.path.join(CORPORA_FOLDER, 'plots')
    DOC_FOLDER = os.path.join(DOC_PLOTS, 'docs')
    DOC_DIGRAMS = os.path.join(DOC_FOLDER, 'digrams')

    # Go through all filenames in the diagrams and unigrams directory
    for filename in os.listdir(DOC_DIGRAMS):
        name = filename
        language = name.split("_")[0]
        full_language = [y for x, y in languages_lib if x == language][0]
        caption = r"\small{" + \
            f"{full_language.capitalize()} passages, top starting-digrams." + "}"
        label = f"plot:polar:passages:digrams:{language}"
        res = "\\begin{subfigure}[b]{0.3\\textwidth}\n\t\\centering\n\t\\includegraphics[width=\\textwidth]{images/plots/documents/digrams/" + \
            name + "}\n\t\\caption{\\footnotesize{" + caption + \
            "}}\n\t\\label{fig:" + label + "}\n\\end{subfigure}"
        print(res)


def get_latex_corpora_stats():
    """
    For all files in the corpora stats folder, which follow the following structure (example):
    "language": "ar",
    "passages": 2061414,
    "documents": 656982,
    "mean_document_length": 950.5924072265625,
    "mean_title_length": 15.695849418640137,
    "avg_document_length": 950.5923891369563,
    "avg_title_length": 15.695850033036391,
    "min_document_length": 1,
    "min_title_length": 1,
    "max_document_length": 214076,
    "max_title_length": 129,

    We want to print a LaTeX table with the following structure:
    (Note: any float value should be rounded to 1 decimal place)

                passages | documents | avg doc length | avg title length | max doc length | max title length
    languages
    ar
    bn
    ...

    This function prints the tabularx LaTeX code such that it can easily be copied into a LaTeX document.
    """
    print("\\begin{tabularx}{\\textwidth}{|l|*{7}{X|}}")
    print("\\hline")
    print(r"Code & Language & No. of \newline documents & No. of \newline passages & Avg. length \newline document & Max length \newline document & Avg. length \newline document title & Max length \newline document title \\ \hline")
    rows = []
    for filename in os.listdir(CORPORA_FOLDER):
        if not filename.endswith(".json"):
            continue
        data = json.load(open(os.path.join(CORPORA_FOLDER, filename)))
        language = data["language"].strip()
        full_language = [y for x, y in languages_lib if x ==
                         language][0].capitalize()
        passages = data["passages"]
        documents = data["documents"]
        avg_doc_length = round(data["avg_document_length"], 1)
        avg_title_length = round(data["avg_title_length"], 1)
        max_doc_length = data["max_document_length"]
        max_title_length = data["max_title_length"]

        # Convert numeric string to include dots . for each thousand and for decimals use a comma ,
        # Example: 1000000.3 -> 1.000.000,3
        # Example 2: 17 -> 17
        passages = f"{passages:,}".replace(",", ".")
        documents = f"{documents:,}".replace(",", ".")
        max_doc_length = f"{max_doc_length:,}".replace(",", ".")
        max_title_length = f"{max_title_length:,}".replace(",", ".")
        avg_doc_length = str(avg_doc_length).replace(".", ",")
        avg_title_length = str(avg_title_length).replace(".", ",")
        rows.append(
            f"{language} & {full_language} & {documents} & {passages} & {avg_doc_length} & {max_doc_length} & {avg_title_length} & {max_title_length} \\\\")
    rows.sort()
    for row in rows:
        print(row)
    print("\\hline")
    print("\\end{tabularx}")


def get_latex_queries_stats(split: str = "total"):
    """
    For all files in the queries stats folder, which follow the following structure (example):
        {
            "language": "en",
            "train": {
                "split": "train",
                "queries": 2863,
                "mean_query_length": 39.07125473022461,
                "min_query_length": 11,
                "max_query_length": 129
            },
            "dev": {
                "split": "dev",
                "queries": 799,
                "mean_query_length": 40.24781036376953,
                "min_query_length": 16,
                "max_query_length": 122
            },
            "testA": {
                "split": "testA",
                "queries": 734,
                "mean_query_length": 40.48773956298828,
                "min_query_length": 16,
                "max_query_length": 108
            },
            "testB": {
                "split": "testB",
                "queries": 1790,
                "mean_query_length": 49.61843490600586,
                "min_query_length": 13,
                "max_query_length": 165
            },
            "total": {
                "split": "total",
                "queries": 6186,
                "mean_query_length": 42.443260192871094,
                "min_query_length": 11,
                "max_query_length": 165
            }
        }

    We want to print a LaTex table with the following structure:
    (Note: any float value should be rounded to 1 decimal place)

    Code | Language | no. of queries | avg query length | min query length max query length    
    ar
    bn
    de
    ...

    The splits can be train, dev, testA, testB, total. Note, that not all languages have all splits.
    This function prints the tabularx LaTeX code such that it can easily be copied into a LaTeX document.

    Make a table for each split based on the data in the queries stats folder.

    """
    print("\\begin{tabularx}{\\textwidth}{|l|*{5}{X|}}")
    print("\\hline")
    print(r"Code & Language & No. of queries & Avg. query length & Min. query length & Max. query length \\ \hline")
    rows = []
    for filename in os.listdir(QUERIES_FOLDER):
        if not filename.endswith(".json"):
            continue
        data = json.load(open(os.path.join(QUERIES_FOLDER, filename)))
        language = data["language"].strip()
        full_language = [y for x, y in languages_lib if x ==
                         language][0].capitalize()
        if split not in data:
            continue
        queries = data[split]["queries"]
        avg_query_length = round(data[split]["mean_query_length"], 1)
        min_query_length = data[split]["min_query_length"]
        max_query_length = data[split]["max_query_length"]

        # Convert numeric string to include dots . for each thousand and for decimals use a comma ,
        # Example: 1000000.3 -> 1.000.000,3
        # Example 2: 17 -> 17
        queries = f"{queries:,}".replace(",", ".")
        max_query_length = f"{max_query_length:,}".replace(",", ".")
        avg_query_length = str(avg_query_length).replace(".", ",")
        rows.append(
            f"{language} & {full_language} & {queries} & {avg_query_length} & {min_query_length} & {max_query_length} \\\\")
    rows.sort()
    for row in rows:
        print(row)
    print("\\hline")
    print("\\end{tabularx}")


def language_distribution_plot():
    # Define the data
    data = {
        'Code': ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'yo', 'zh'],
        'Language': ['Arabic', 'Bengali', 'German', 'English', 'Spanish', 'Persian', 'Finnish', 'French', 'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Russian', 'Swahili', 'Telugu', 'Thai', 'Yoruba', 'Chinese'],
        'No. of documents': [656982, 63762, 2651352, 5758285, 1669181, 857827, 447815, 2325608, 148107, 446330, 1133444, 437373, 1476045, 47793, 66353, 128179, 33094, 1246389]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by 'No. of documents' in descending order
    df = df.sort_values(by='No. of documents', ascending=False)

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x='No. of documents', y='Code', orient='h')

    # Colormap the bars
    cmap = cm.get_cmap('summer')
    colors = [cmap(i/len(df)) for i in range(len(df))]
    for i, bar in enumerate(plt.gca().patches):
        bar.set_color(colors[i])

    # Add labels and title
    plt.xlabel('Number of documents (in millions)', fontsize=20)
    plt.ylabel('Language Code', fontsize=20)

    # Set x axis to more frequent ticks and vertical labels with the full number
    plt.xticks(rotation=90)
    plt.xticks(np.arange(0, 6000000, 250000))

    # Gridlines (behind the bars)
    plt.grid(True, axis='x')

    # Set grdlines behind the bars like: ax.set_axisbelow(True)
    # Get ax
    ax = plt.gca()
    ax.set_axisbelow(True)

    # set font size to 20
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=20)

    # set fig size
    plt.gcf().set_size_inches(16, 10)

    # Show the plot
    plt.tight_layout()
    plt.show()


def language_distribution_plot2():
    # Define the data
    data = {
        'Code': ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'yo', 'zh'],
        'Language': ['Arabic', 'Bengali', 'German', 'English', 'Spanish', 'Persian', 'Finnish', 'French', 'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Russian', 'Swahili', 'Telugu', 'Thai', 'Yoruba', 'Chinese'],
        'No. of documents': [656982, 63762, 2651352, 5758285, 1669181, 857827, 447815, 2325608, 148107, 446330, 1133444, 437373, 1476045, 47793, 66353, 128179, 33094, 1246389],
        'Avg. length document title': [15.7, 16.2, 18.8, 19.9, 19.6, 14.3, 16.2, 19.7, 15.5, 18.6, 8.2, 7.3, 21.2, 13.5, 14.5, 18.7, 15.7, 6.3]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by 'No. of documents' in descending order
    df = df.sort_values(by='Avg. length document title', ascending=False)

    # Create the bar plot for average title length
    plt.figure(figsize=(14, 8))
    ax2 = sns.barplot(data=df, x='Avg. length document title',
                      y='Code', orient='h')

    # Colormap the bars
    cmap = cm.get_cmap('winter')
    colors = [cmap(i/len(df)) for i in range(len(df))]
    for i, bar in enumerate(ax2.patches):
        bar.set_color(colors[i])

    # Add labels and title
    plt.xlabel('Average Length of Document Title', fontsize=20)
    plt.ylabel('Language Code', fontsize=20)

    # Set x-axis to more frequent ticks and vertical labels with the full number
    plt.xticks(rotation=0, fontsize=15)
    plt.xticks(np.arange(0, 25, 1))

    # Set fontsize y labels
    plt.yticks(fontsize=20)

    # Gridlines (behind the bars)
    plt.grid(True, axis='x')

    # Set gridlines behind the bars
    ax2.set_axisbelow(True)

    # Show the plot
    plt.tight_layout()
    plt.show()


def query_distribution_plot():
    data = {
        'Code': ['ar', 'bn', 'de', 'en', 'es', 'fa', 'fi', 'fr', 'hi', 'id', 'ja', 'ko', 'ru', 'sw', 'te', 'th', 'yo', 'zh'],
        'Language': ['Arabic', 'Bengali', 'German', 'English', 'Spanish', 'Persian', 'Finnish', 'French', 'Hindi', 'Indonesian', 'Japanese', 'Korean', 'Russian', 'Swahili', 'Telugu', 'Thai', 'Yoruba', 'Chinese'],
        'No. of queries': [8732, 3274, 1017, 6186, 4325, 4215, 5939, 2287, 2338, 6373, 6128, 2761, 7564, 3486, 5667, 5347, 407, 2625],
        'Avg. query length': [29.9, 45.1, 44.9, 42.4, 47.1, 40.8, 37.7, 43.9, 52.5, 36.9, 17.3, 19.7, 43.5, 39.8, 37.7, 41.6, 37.5, 11.0],
        'Min. query length': [11, 15, 12, 11, 15, 15, 13, 14, 19, 12, 5, 4, 14, 12, 12, 11, 19, 6],
        'Max. query length': [101, 132, 95, 165, 92, 116, 149, 103, 120, 128, 57, 142, 184, 101, 119, 185, 69, 28]
    }

    # Convert data to DataFrame
    df = pd.DataFrame(data)

    # Sort DataFrame by 'Avg. query length' in descending order
    df = df.sort_values(by='Avg. query length', ascending=True)

    # Set the style of seaborn
    sns.set_theme(style="whitegrid")

    # Create subplots
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

    # Plot average query length
    sns.barplot(ax=axes[0], data=df, x='Avg. query length',
                y='Language', palette='viridis')
    axes[0].set_title('Average Query Length', fontsize=20)
    axes[0].set_xticks(np.arange(0, 61, 5))

    # Plot minimum query length
    sns.barplot(ax=axes[1], data=df, x='Min. query length',
                y='Language', palette='viridis')
    axes[1].set_title('Minimum Query Length', fontsize=20)

    # Set min query length x ticks to more frequent ticks
    axes[1].set_xticks(np.arange(0, 22, 2))

    # Plot maximum query length
    sns.barplot(ax=axes[2], data=df, x='Max. query length',
                y='Language', palette='viridis')
    axes[2].set_title('Maximum Query Length', fontsize=20)
    axes[2].set_xticks(np.arange(0, 201, 25))

    # Set fontsize x labels and y labels
    for ax in axes:
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis='x', labelsize=16)
        ax.tick_params(axis='y', labelsize=16)

    plt.tight_layout()
    plt.show()


def get_latex_mdpr_table():
    """
    Get the LaTeX code for the table of the mDPR results.
    Meaning, the name contains "mdpr".

    The YAML file is structured as follows (two examples):

    conditions:
        - name: bm25-mdpr-tied-pft-msmarco-hybrid.ja
            eval_key: miracl-v1.0-ja
            command: python -m pyserini.fusion --runs ${bm25_output} ${mdpr_output} --output ${output} --method interpolation --alpha 0.5 --depth 1000 --k 1000
            splits:
            - split: train
                scores:
                - nDCG@10: 0.5795
                    R@100: 0.9082
            - split: dev
                scores:
                - nDCG@10: 0.5757
                    R@100: 0.9036
        - name: bm25-mdpr-tied-pft-msmarco-hybrid.ko
            eval_key: miracl-v1.0-ko
            command: python -m pyserini.fusion --runs ${bm25_output} ${mdpr_output} --output ${output} --method interpolation --alpha 0.5 --depth 1000 --k 1000
            splits:
            - split: train
                scores:
                - nDCG@10: 0.5758
                    R@100: 0.8744
            - split: dev
                scores:
                - nDCG@10: 0.6086
                    R@100: 0.8997

    The language codes are the same as in the previous functions.
    They can be found in the "name", splitting on . and taking the last element.

    Get all the results for the "train" and "dev" splits for mDPR and print them in a LaTeX table.
    Note: the languages should be sorted alphabetically.

    The final table should look like

    Model    Metric     Split                        Languages
                                 ar       bn      de ...

              nDCG@10   Train    [score] [score] [score] ...
    mdpr..              Dev
              R@100     Train
                        Dev

    """
    results_file = os.path.join(STATS_FOLDER, 'results_bm25_mdpr.yaml')
    results = yaml.load(open(results_file), Loader=yaml.FullLoader)

    parsed_data = results

    # Extract relevant information and organize into a dictionary
    results = {}
    for entry in parsed_data['conditions']:
        model = entry['name'].split('.')[0]
        language = entry['name'].split('.')[-1]
        for split in entry['splits']:
            split_name = split['split']
            for score in split['scores']:
                metric = list(score.keys())[0]
                value = score[metric]
                if 'mdpr' in model:
                    key = (model, metric, split_name)
                    if key not in results:
                        results[key] = {}
                    results[key][language] = value

    # Sort languages alphabetically for each model
    for key in results:
        results[key] = dict(sorted(results[key].items()))

    # Print LaTeX table
    print("\\begin{table}[H]")
    print("\centering")
    print("\\begin{tabular}{|c|c|c|c|c|c|}")
    print("\hline")
    print("Model & Metric & Split & Languages \\\\")
    print("\hline")
    for key, value in results.items():
        model, metric, split = key
        languages = " & ".join([f"{lang} [{value[lang]}]" for lang in value])
        print(f"{model} & {metric} & {split} & {languages} \\\\")
        if metric == 'nDCG@10' and split == 'Dev':
            print("\\hline")
    print("\hline")
    print("\end{tabular}")
    print("\caption{Your caption here}")
    print("\label{tab:yourlabel}")
    print("\end{table}")


if __name__ == "__main__":
    get_latex_mdpr_table()
