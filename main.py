from utils import do_stuff_and_get_summary, load_data_from_pickle

SUMMARY_LENGTH = 5


if __name__ == '__main__':
    train_corpus_path = "data/log_files/train.txt"
    data_map = load_data_from_pickle(train_corpus_path)
    list_of_sentences, sentence_metadata = get_sentences_with_metadata(data_map)
    summary = do_stuff_and_get_summary(list_of_sentences, sentence_metadata)
    print(summary)
