
SUMMARY_LENGTH = 5


if __name__ == '__main__':

    list_of_sentences, sentence_metadata = get_sentences_with_metadata(data_map)
    summary = do_stuff_and_get_summary(list_of_sentences, sentence_metadata)
    list_of_sentences, sentence_metadata = get_sentences_with_metadata(data_map)
    list_of_sentences = [sentence.strip() for sentence in list_of_sentences if len(sentence) > 1]
    processed_sentences = make_processed_sentences(list_of_sentences)
    sentence_graph, sentence_common_graph = make_graph(processed_sentences, sentence_metadata)
    sentence_scores = calculate_scores(sentence_graph)
    sentence_page_scores = calculate_pagerank_scores(sentence_common_graph)
    sentence_score_final = [sentence_scores[i] * (sentence_page_scores[i]+1)  for i in range(len(sentence_scores))]


    ### Couple of different ways of doingsummaries.
    summary = rank_sentences_and_make_summary(list_of_sentences, processed_sentences, sentence_graph, sentence_scores, SUMMARY_LENGTH)
    summary = rank_sentences_and_make_summary(list_of_sentences, processed_sentences, sentence_graph, sentence_score_final, SUMMARY_LENGTH)
    summary = rank_sentences_and_make_summary(list_of_sentences, processed_sentences, sentence_graph, sentence_page_scores, SUMMARY_LENGTH)
