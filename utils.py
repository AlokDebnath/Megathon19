from summarization_utils import *


def do_stuff_and_get_summary(list_of_sentences, sentence_metadata):
    list_of_sentences = [sentence.strip() for sentence in list_of_sentences if len(sentence) > 1]
    processed_sentences = make_processed_sentences(list_of_sentences)
    sentence_graph, sentence_common_graph = make_graph(processed_sentences, sentence_metadata)
    sentence_scores = calculate_scores(sentence_graph)
    sentence_page_scores = calculate_pagerank_scores(sentence_common_graph)
    sentence_score_final = [sentence_scores[i] * (sentence_page_scores[i]+1)for i in range(len(sentence_scores))]
    summary_length = 5
    summary = rank_sentences_and_make_summary(list_of_sentences, processed_sentences, sentence_graph, sentence_score_final, summary_length)
    return summary

def load_data_from_pickle(train_corpus_path):
    data_path = train_corpus_path.rsplit('/', 1)[0] + "/SplitTrain"
    data_map = {}
    with open(data_path + "/" + "train_0.pickle", 'rb') as handle:
        data_map = pickle.load(handle)
    return data_map
