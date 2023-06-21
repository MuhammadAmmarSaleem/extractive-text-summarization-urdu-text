"""
Container for all sentence metrics
"""
import math


def doc_aware_pos_ratios(features, sentence):
    nn_count = 0
    ve_count = 0
    aj_count = 0
    av_count = 0
    num_count = 0

    for index, word_tuple in enumerate(sentence):
        pos = word_tuple[1]
        if pos == "NN":
            nn_count += 1
        elif pos == "VB":
            ve_count += 1
        elif pos == "ADJ":
            aj_count += 1
        elif pos == "ADV":
            av_count += 1
        elif pos == "OR" or pos == "FR" or pos == "MUL":
            num_count += 1

    features['doc_aware_noun_ratio'] = nn_count / features['doc_nouns']
    features['doc_aware_verb_ratio'] = ve_count / features['doc_verbs']
    features['doc_aware_adj_ratio'] = aj_count / features['doc_adjcs']
    features['doc_aware_adv_ratio'] = av_count / features['doc_advbs']
    features['doc_aware_num_ratio'] = num_count / features['doc_nums']


def doc_unaware_pos_ratio(features, sentence):
    nn_count = 0
    ve_count = 0
    aj_count = 0
    av_count = 0
    num_count = 0

    for index, word_tuple in enumerate(sentence):
        pos = word_tuple[1]
        if pos == "NN":
            nn_count += 1
        elif pos == "VB":
            ve_count += 1
        elif pos == "ADJ":
            aj_count += 1
        elif pos == "ADV":
            av_count += 1
        elif pos == "OR" or pos == "FR" or pos == "MUL":
            num_count += 1
    try:
        features['doc_unaware_noun_ratio'] = nn_count / len(sentence) if len(sentence) > 0 else 0
        features['doc_unaware_verb_ratio'] = ve_count / len(sentence) if len(sentence) > 0 else 0
        features['doc_unaware_adj_ratio'] = aj_count / len(sentence) if len(sentence) > 0 else 0
        features['doc_unaware_adv_ratio'] = av_count / len(sentence) if len(sentence) > 0 else 0
        features['doc_unaware_num_ratio'] = num_count / len(sentence) if len(sentence) > 0 else 0
    except:
        print("Noun Count: ", nn_count)
        print("Verb Count: ", ve_count)
        print("Adj Count: ", aj_count)
        print("Adv Count: ", av_count)
        print("Num Count: ", num_count)


def frequency_score(sentence_words, word_freq):
    """
    Term Frequency measure, average
    Args:
        sentence: An array of tokenized words of the sentence
    """
    sen_score = 0
    for index, sen_word_tuple in enumerate(sentence_words):
        word = sen_word_tuple[0]
        sen_score = sen_score + word_freq[word]
    return sen_score / len(sentence_words)


def inverse_sentence_freq(term, text_sents_tokenized):
    """
    Computes ISF
    Args:
        term: the word for which isf will be calculated
        sentences: array of all sentences in the text, tokenized and removed stop words
    """
    sentences_containing = 0
    for sen in text_sents_tokenized:
        if term in sen:
            sentences_containing = sentences_containing + 1
    if sentences_containing == 0:
        sentences_containing = 1
    return 1 - (math.log(sentences_containing) / math.log(len(text_sents_tokenized)))


def inverse_sentence_freq_old(term, sentences):
    """
    Computes ISF
    Until 06/26/2020, this variation of isf was used in our code, however I couldn't remember why we had devised such a
    formula and I couldn't find it in any reference! so I suffixed it with _old and put here for later reference.
    A more standard formula is used now in inverse_sentence_freq()
    Args:
        term: the word for which isf will be calculated
        sentences: array of all sentences in the text, tokenized and removed stop words
    """
    sentences_containing = 0
    for sen in sentences:
        if term in sen:
            sentences_containing = sentences_containing + 1
    if sentences_containing == 0:
        sentences_containing = 1
    return 1 - (math.log(sentences_containing) / math.log(len(sentences)))


def find_relative_length(text_sents_tokenized, sent):
    sent_length = len(sent)

    # find average length of sentence in document
    total_words_doc = find_doc_words(text_sents_tokenized)
    avg_sent_length = total_words_doc / len(text_sents_tokenized)

    relative_length = sent_length / avg_sent_length
    return relative_length


def tf_isf_score(sen, text_sents_tokenized, word_freq):
    sen_score = 0
    for index, sen_word_tuple in enumerate(sen):
        word = sen_word_tuple[0]
        sen_score = sen_score + word_freq[
            word] * inverse_sentence_freq(word, text_sents_tokenized)

    return sen_score / len(sen) if len(sen) > 0 else 0


def linear_poition_score(position, total_sentences):
    return 1 - (position / total_sentences)


def cosine_position_score(position, total_sentences):
    alpha = 2

    return (math.cos(
        (2 * 3.14 * position) / (total_sentences - 1)) + alpha - 1) / alpha



def title_similarity_score(sen, title):
    denominator = math.sqrt(len(sen) * len(title))
    if denominator > 0:
        ratio = len(set(sen).intersection(title)) / denominator
    else:
        ratio = 0
    return ratio


def cosine_similarity(vec1, vec2):
    """
    Computes cosine similarity of the given parameters
    Args:
        vec1: frequency distribution of a sentence
        vec2: frequency distribution of a sentence
    """
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])

    sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
    sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)

    if not denominator:
        return 0.0
    else:
        return float(numerator) / denominator


def cue_words(sentence_words, cue_words_list):
    '''

    '''
    output = 0
    for word in sentence_words:
        if word in cue_words_list:
            output += 1
    return output


def find_doc_words(text_sents_tokenized):
    total_doc_words = 0;
    for sen in text_sents_tokenized:
        total_doc_words += len(sen)

    return total_doc_words


def find_cue_phrases(ur_cue_words, sent_text):
    num_of_cue_words = 0
    for cue_word in ur_cue_words:
        num_of_cue_words += sent_text.count(cue_word)

    return num_of_cue_words
