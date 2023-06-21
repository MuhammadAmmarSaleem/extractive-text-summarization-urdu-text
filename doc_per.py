import hazm
import hashlib
from utilities import *
import time
from nltk.probability import FreqDist
import urduhack
from urduhack import normalize
from urduhack.tokenization import sentence_tokenizer
from urduhack.tokenization import word_tokenizer

"""
normalizer = hazm.Normalizer()
stemmer = hazm.Stemmer()
tagger = hazm.POSTagger(model='resources/postagger.model')
"""


def document_features_set(text_sents_tokenized, text_sents_pos, ur_cue_words, text_sents, category, golden_summaries=[],
                         key=''):
    feature_set = []
    position = 1
    all_words = get_word_tokens(text_sents_tokenized)
    word_freq = FreqDist(all_words)
    doc_pos_features = find_doc_pos(text_sents_pos)
    for sen_index, sen in enumerate(text_sents_pos):
        features = generate_features(sen, text_sents_pos, text_sents_tokenized, word_freq, position,
                                     doc_pos_features, category, ur_cue_words, text_sents[sen_index])
        feature_set.append(features)
        position += 1
    return feature_set
