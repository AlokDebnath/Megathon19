import os
import sys
import json
import pickle
from tqdm import tqdm
from nltk.corpus import stopwords
import nltk
import operator
import numpy as np
import pandas as pd
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models import Word2Vec


model = gensim.models.KeyedVectors.load_word2vec_format('data/glove.6B.300d.w2vformat.txt')

word_freq_map = {}
with open("data/arxiv-release/vocab", 'r') as vocab_file:
    lines = vocab_file.readlines()
    for line in lines:
        word_freq_map[line.split()[0]] = int(line.split()[1])

stop_list = sorted(word_freq_map.items(), key=operator.itemgetter(1), reverse=True)[:150]
cache = {}


def read_data_and_split(train_corpus_path):
    train_data_map = {}
    file_no = 0
    with open(train_corpus_path, 'r') as train_data_file:
        line_count = 0
        while file_no < 11:
            if line_count < 20000:
                line_data = train_data_file.readline()
                if line_data:
                    line_map = json.loads(line_data)
                    article_id = line_map['article_id']
                    del line_map['article_id']
                    train_data_map[article_id] = line_map
                    line_count += 1
                else:
                    break
            else:
                with open(train_corpus_path.rsplit('/', 1)[0] + "/SplitTrain/" + "train_" + str(file_no) + ".pickle", 'wb') as train_file:
                    pickle.dump(train_data_map, train_file, protocol=pickle.HIGHEST_PROTOCOL)
                print("File ", file_no, " Done")
                train_data_map.clear()
                file_no += 1
                line_count = 0
        with open(train_corpus_path.rsplit('/', 1)[0] + "/SplitTrain/" + "train_" + str(file_no) + ".pickle", 'wb') as train_file:
            pickle.dump(train_data_map, train_file, protocol=pickle.HIGHEST_PROTOCOL)


def get_sentences_with_metadata(data_map):
    full_text = []
    sentence_metadata = []
    list_of_sentences = []
    summary_list = []
    abstract_list = []
    c = 0
    file_number = 1
    for article_id, data in tqdm(data_map.items()):
        c += 1
        if c == 13:
            abstract_list.append(data['abstract_text'])
            section_data = data['sections']
            section_names = data['section_names']
            for i, section in enumerate(section_data):
                for line in section:
                    split_line = line.split('.')
                    for l in split_line:
                        list_of_sentences.append(l)
                        sentence_metadata.append(section_names[i])
            break
            summary_list.append(do_stuff_and_get_summary(list_of_sentences, sentence_metadata))
            list_of_sentences.clear()

        else:
            continue
            c = 0
            write_summary_and_abstract_to_file(summary_list, abstract_list, file_number)
            print("File ", file_number, " done")
            file_number += 1
            summary_list.clear()
            abstract_list.clear()
    print(list_of_sentences)
    return list_of_sentences, sentence_metadata


def is_ascii(word):
    """
    Checks if word is ascii or not
    :param word: token
    :return: Boolean
    """
    valid = True
    try:
        word = word.encode('ascii')
    except UnicodeEncodeError:
        valid = False
    return valid


def get_processed_tokens(sentence):
    punc_map = {}
    punc_map = punc_map.fromkeys('!"\'()*+,;<>[]^`{|}~:=%&_#?-$/', ' ')
    table = str.maketrans(punc_map)
    tokens = sentence.lower().translate(table).split()
    stop_words = set(stopwords.words('english'))
    stop_words = list(stop_words) + stop_list
    cleaned_tokens = [word for word in tokens if word not in stop_words and is_ascii(word) and '@' not in word and '\\' not in word and len(word) > 1]
    return cleaned_tokens


def make_processed_sentences(list_of_sentences):
    processed_sentences = []
    for sentence in list_of_sentences:
        if isinstance(sentence, list):
            sentence = " ".join(sentence)
        processed_sentences.append(get_processed_tokens(sentence))
    return processed_sentences


def get_no_of_common_word(sentence1, sentence2):
    common_count = 0
    for s1 in sentence1:
        for s2 in sentence2:
            if s1 == s2:
                common_count += 1
    return common_count


def get_word_vec_sim(sentence1, sentence2):
    score = 0
    for word1 in sentence1:
        for word2 in sentence2:
            try:
                temp = cache[word1+word2]
            except:
                try:
                    temp = model.similarity(word1, word2)
                    cache[word1+word2] = temp
                    cache[word2+word1] = temp
                except:
                    cache[word1+word2] = 0
                    cache[word2+word1] = 0
                    temp = 0
            score += temp
    return score


def scoring(sentence1, sentence2, metadata):
    len_normalize = len(sentence1) + len(sentence2) + 1 # Normalizing by length of vector
    common_words = get_no_of_common_word(sentence1, sentence2)
    word_vec_score = get_word_vec_sim(sentence1, sentence2)
    score = common_words / 2*len_normalize + word_vec_score / len_normalize
    return score


def make_graph(processed_sentences, metadata):
    sentence_graph = np.zeros(shape=(len(processed_sentences), len(processed_sentences)))
    sentence_common_graph = np.zeros(shape=(len(processed_sentences), len(processed_sentences)))

    for i in range(len(processed_sentences)):
        for j in range(len(processed_sentences)):
            sentence1 = processed_sentences[i]
            sentence2 = processed_sentences[j]
            if i == j:
                sentence_graph[i][j] = 0
                sentence_common_graph[i][j] = 0
            else:
                sentence_graph[i][j] = scoring(sentence1, sentence2, metadata)
                sentence_common_graph[i][j] = get_no_of_common_word(sentence1, sentence2)
    return sentence_graph, sentence_common_graph

# Aggregation
def calculate_scores(sentence_graph):
    scores = np.zeros(len(sentence_graph))
    for i,sentence in enumerate(sentence_graph):
        scores[i] = sum(sentence_graph[i])
    return scores


# Page Rank
def calculate_pagerank_scores(sentence_graph):
    N = len(sentence_graph)
    d = 0.15   # PageRank Hyperparameter
    pagerank_scores = np.ones(N)

    out_degree = np.zeros(N)
    for i in range(N):
        for j in range(N):
            if sentence_graph[i][j]:
                out_degree[i] += sentence_graph[i][j]

    for i in range(N):
        score = 0
        for j in range(N):
            if sentence_graph[j][i]:
                score += (pagerank_scores[j] / out_degree[j])
        pagerank_scores[i] = (d / N) + (1 - d) * score
    return pagerank_scores


def rank_sentences_and_make_summary2(sentences, processed_sentences, sentence_graph, scores):
    scores_indices = np.argsort(scores)
    ordered_sentences = scores_indices[::-1]
    summary = []
    for i in range(5):
        summary.append(sentences[ordered_sentences[i]])
#         print(ordered_sentences[i], scores[ordered_sentences[i]])
#         print(processed_sentences[ordered_sentences[i]])
    return summary


def rank_sentences_and_make_summary(sentences, processed_sentences, sentence_graph, scores, summary_length):
    summary = []
    for i in range(summary_length): # Number of Sentences we want in the summary
        score_indices = np.argsort(scores)
        if len(score_indices) < 1:
            break
        selected_index = score_indices[-1]
        summary.append(sentences[selected_index]) # Adding highest score sentence to summary
        mean_score = np.mean(sentence_graph)
        to_decrease = []
        # Calculated mean similarity score. If selected sentence and another sentence have
        # high similarity, the score of the second sentence should be reduced.
        # Here, have chosen to use 1.5 * mean_score as the threshold, and divided score in half.
        for iterator in range(len(processed_sentences)):
            if sentence_graph[iterator][selected_index] > 1.5 * mean_score:
                to_decrease.append(iterator)
            if sentence_graph[selected_index][iterator] > 1.5 * mean_score:
                to_decrease.append(iterator)
        for sentence in set(to_decrease):
            # Should be changed based on the number of sentences needed in the summary
            scores[sentence] /= (1 + 1.0 / summary_length) # Reduced score by half, to on average prevent from being picked.
        scores[selected_index] = 0
    return summary

