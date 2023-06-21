import json, random, re, hashlib
import os

# from hazm import *
import nltk, math, operator
from nltk import bleu
from fractions import Fraction
import rouge
import Features

"""
all_features = ['position', 'cosine_position', 'cue_words', 'tfisf', 'tf', 'len', 'relative_len', 'num_count',
                'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio', 'doc_words',
                'doc_sens', 'doc_parag',
                'included', 'target', 'target_bleu_avg', 'text', 'target_bleu', 'source_file', 'id', 'category',
                'doc_verbs', 'doc_adjcs', 'doc_advbs', 'doc_nouns', 'doc_nums', 'nnf_isnnf', 'vef_isvef', 'ajf_isajf',
                'avf_isavf', 'nuf_isnuf', 'political', 'social', 'sport', 'culture', 'economy', 'science'
                ]

learning_features = ['position', 'cosine_position', 'cue_words', 'tfisf', 'tf', 'len', 'relative_len', 'num_count',
                     'pos_ve_ratio', 'pos_aj_ratio', 'pos_nn_ratio', 'pos_av_ratio', 'pos_num_ratio',
                     'doc_words', 'doc_sens', 'doc_parag', 'category', 'doc_verbs', 'doc_adjcs', 'doc_advbs',
                     'doc_nouns', 'doc_nums', 'nnf_isnnf', 'vef_isvef', 'ajf_isajf', 'avf_isavf', 'nuf_isnuf',
                     'political', 'social', 'sport', 'culture', 'economy', 'science'
                     ]
"""

learning_features = ["ordinal_position", "length_of_sentence", "doc_unaware_noun_ratio", "doc_unaware_verb_ratio",
                     "doc_unaware_adj_ratio",
                     "doc_unaware_adv_ratio",
                     "cue_words",
                     "doc_unaware_num_ratio", "explicit_doc_sents",
                     "explicit_doc_words",
                     "explicit_topical_categoty", "doc_aware_cosine_position", "doc_aware_relative_length",
                     "doc_aware_noun_ratio",
                     "doc_aware_verb_ratio", "doc_aware_adj_ratio", "doc_aware_adv_ratio", "doc_aware_num_ratio",
                     "doc_aware_tfisf"]

document_aware = ["doc_aware_cosine_position", "doc_aware_relative_length", "doc_aware_noun_ratio",
                  "doc_aware_verb_ratio", "doc_aware_adj_ratio", "doc_aware_adv_ratio", "doc_aware_num_ratio",
                  "doc_aware_tfisf"]

explicit_doc_features = ["explicit_doc_sents", "explicit_doc_words", "explicit_topical_categoty"]

document_unaware = ["ordinal_position", "length_of_sentence", "doc_unaware_noun_ratio", "doc_unaware_verb_ratio",
                    "doc_unaware_adj_ratio",
                    "doc_unaware_adv_ratio",
                    "cue_words",
                    "doc_unaware_num_ratio"]

all_feature_names = document_aware + explicit_doc_features + document_unaware

experiment1_feature_names = document_unaware
experiment2_feature_names = explicit_doc_features + document_aware
experiment3_feature_names = all_feature_names

# selected_feature_names = experiment2_feature_names
selected_feature_names = document_aware


def corpures_category_mapping(category):
    category_map = {
        'CurrentAfr': 1,
        'Ent': 2,
        'Health': 3,
        'Int': 4,
        'Sport': 5,

    }
    return category_map[category]


similarity_threshold = 0.45


def read_file(path):
    file = open(path, "r", encoding='utf8')
    content = file.read()
    return content


def write_file(path, data):
    file = open(path, "w+")
    output = file.write(data)
    file.close()
    return output


def load_dataset(path):
    dataset = json.loads(read_file(path))
    features = []
    target = []
    labels = []
    for key in dataset:
        for (sen, label) in dataset[key]:
            row = []
            for attr in learning_features:
                row.append(sen[attr])

            features.append(row)
            target.append(sen['target'])
            labels.append(label)
    return features, target, labels


def select_features(feature_names, matrix):
    feature_indexes = []
    for name in feature_names:
        if name in learning_features:
            col_index = learning_features.index(name)
            feature_indexes.append(col_index)
    features = matrix[:, feature_indexes]
    return features


def balance_dataset(feature_vectors, targets, labels, ratio):
    num_true = 0
    num_false = 0
    false_indices = []
    balanced_x_train = []
    balanced_y_train = []
    balanced_labels = []
    for i in range(len(feature_vectors)):
        if targets[i] > similarity_threshold:
            num_true += 1
            balanced_x_train.append(feature_vectors[i])
            balanced_y_train.append(targets[i])
            balanced_labels.append(labels[i])
        else:
            num_false += 1
            false_indices.append(i)
    print("Number of positives/negatives: {}/{} ".format(num_true, num_false))
    selected_indices = random.sample(false_indices, int(num_true * ratio))
    print("After balancing, positives/negatives: {}/{} ".format(num_true, len(selected_indices)))
    for i in selected_indices:
        balanced_x_train.append(feature_vectors[i])
        balanced_y_train.append(targets[i])
        balanced_labels.append(labels[i])
    return balanced_x_train, balanced_y_train, balanced_labels


def normalize_dataset(feature_matrix, feature_names, mode='utilization'):
    if normalize_dataset.scalers is None:
        from sklearn.preprocessing import MinMaxScaler
        scalers = {
            # 'doc_words': MinMaxScaler(),
            # 'doc_nouns': MinMaxScaler(),
            # 'doc_verbs': MinMaxScaler(),
            # 'doc_adjcs': MinMaxScaler(),
            # 'doc_advbs': MinMaxScaler(),
            'explicit_doc_sents': MinMaxScaler(),
            'doc_aware_tfisf': MinMaxScaler(),
            'cue_words': MinMaxScaler(),
            'length_of_sentence': MinMaxScaler(),
            'doc_aware_relative_length': MinMaxScaler(),
        }
        normalize_dataset.scalers = scalers
    elif mode == 'learn':
        raise Exception('normalize_dataset must not be called again in learn mode')
    else:
        scalers = normalize_dataset.scalers

    for feature_name in scalers:
        if feature_name in feature_names:
            scaler = scalers[feature_name]
            col_index = feature_names.index(feature_name)
            col = feature_matrix[:, col_index].reshape(-1, 1)
            if mode == 'learn':
                scaler.fit(col)
            # col_transform = scaler.transform(col)
            col_transform = scaler.fit_transform(col)
            feature_matrix[:, col_index] = col_transform.reshape(1, -1)


normalize_dataset.scalers = None


def are_similar_rouge(sen1, sen2):
    scores = rouge.get_scores(sen1, sen2)
    return (scores[0]['rouge-2']['f'] >= 0.7)


def are_similar(sen1, sen2):
    denominator = float(len(set(sen1).union(sen2)))
    if denominator > 0:
        ratio = len(set(sen1).intersection(sen2)) / denominator
    else:
        ratio = 0
    return ratio >= similarity_threshold, ratio


def average_similarity(sen, gold_summaries):
    total_similarity = 0
    for sentence_list in gold_summaries:
        max_similarity = 0
        for sum_sen in sentence_list:
            (similar, similarity) = are_similar(sen, sum_sen)
            if similarity > max_similarity:
                max_similarity = similarity
        total_similarity += max_similarity
    return total_similarity / len(gold_summaries)


def avg_bleu_score(sen, summaries, avg=False):
    min_length = 5
    if avg:
        from nltk.translate.bleu_score import SmoothingFunction
        chencherry = SmoothingFunction()
        total = 0
        for summ in summaries:
            total += bleu([summ], sen, smoothing_function=chencherry.method2)
        score = total / len(summaries)
    else:
        #        score = bleu(summaries, sen, smoothing_function=chencherry.method2)
        score = nltk.translate.bleu_score.modified_precision(summaries, sen, 2)
        if len(sen) < min_length:
            import numpy as np
            score *= np.exp(1 - (min_length / len(sen)))
    return score


def encode_complex(obj):
    if isinstance(obj, Fraction):
        return obj.numerator / obj.denominator
    raise TypeError(repr(obj) + " is not JSON serializable")


def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes
    Thanks to https://gist.github.com/zachguo/10296432
    """
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    # Print header
    print("    " + empty_cell, end=" ")
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.0f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()

    tp = cm[1, 1]
    tn = cm[0, 0]
    tn_percent = 100 * tn / sum(cm[0])
    tp_percent = 100 * tp / sum(cm[1])
    fp = cm[0, 1]
    fn = cm[1, 0]
    total = sum(cm[0]) + sum(cm[1])
    accuracy = 100 * (tp + tn) / total
    precision = 100 * tp / (tp + fp)
    recall = 100 * tp / (tp + fn)
    print('True Positives: {:.2f}%'.format(tp_percent))
    print('True Negatives: {:.2f}%'.format(tn_percent))
    print('Accuracy: {:.2f}%'.format(accuracy))
    print('Precision: {:.2f}%'.format(precision))
    print('Recall: {:.2f}%'.format(recall))


def print_rouges(rouges):
    # print("Diff" + str(diff_summs))
    print("{:<8} {:<25} {:<25} {:<25}".format('Test,', 'f-measure,', 'precision,', 'recall'))
    for k in ['rouge-1', 'rouge-2', 'rouge-l']:
        v = rouges[k]
        print("{:<8}, {:<25}, {:<25}, {:<25}".format(k, v['f'], v['p'], v['r']))


def write_dataset_csv(feats, path):
    first_row = feats[list(feats.keys())[0]][0][0]  # get one of values
    all_columns = sorted(first_row.keys())
    output = [','.join(all_columns) + " \r\n"]

    for key in feats:
        for (sen, target) in feats[key]:
            row = []
            for attr in all_columns:
                row.append(str(sen[attr]))
            output.append(','.join(row) + "\r\n")
            '''str(sen['id']) + "," + str(sen['pos_nn_ratio']) + "," + str(sen['pos_ve_ratio']) + "," +\
            str(sen['pos_aj_ratio']) + "," + str(sen['pos_av_ratio']) + "," + str(sen['tfisf']) + "," + \
            str(sen['tf']) + "," + str(sen['cue_words']) + "," + str(sen['cosine_position']) + "," + \
            str(target)+ "\r\n" )'''

    f_file = open(path, '+w')
    f_file.writelines(output)
    f_file.close()
    print(path + " has been written successfully")


def cue_words(language):
    if language in cue_words.static:
        return cue_words.static[language]
    lang_map = {
        'en': "resources/cue-words-en.txt",
        'fa': "resources/cue-words.txt"
    }

    cue_words.static[language] = read_file(lang_map[language]).split()
    return cue_words.static[language]


cue_words.static = {}


def json_write(data, path):
    import json
    file = open(path, "w+")
    json.dump(data, file, ensure_ascii=False)
    file.close()
    return True


def json_read(path):
    import json
    file = open(path, "r")
    data = json.load(file)
    file.close()
    return data


def export_model(model, export_name):
    import _pickle as cPickle
    with open('models/' + export_name + '.pkl', 'wb') as fid:
        cPickle.dump(model, fid)
    cPickle.dump(normalize_dataset.scalers, open('data/scalers.pkl', 'wb'))


def english_stemmer():
    if english_stemmer.cache:
        return english_stemmer.cache

    from nltk.stem.snowball import SnowballStemmer
    english_stemmer.cache = SnowballStemmer("english")
    return english_stemmer.cache


english_stemmer.cache = None


def cnn_html_escape(text):
    html_escape_table = {
        "&": "&amp;",
        '"': "&quot;",
        "'": "&apost;",
        ">": "&gt;",
        "<": "&lt;",
    }
    return "".join(html_escape_table.get(c, c) for c in text)


def load_features(dataset):
    if load_features.cache:
        return load_features.cache
    import _pickle as cPickle
    load_features.cache = cPickle.load(open('resources/' + dataset + '/features.pkl', 'rb'))
    return load_features.cache


load_features.cache = None


def draw_bar_chart(data, y_label, title):
    import matplotlib.pyplot as plt;
    plt.rcdefaults()
    import numpy as np
    import matplotlib.pyplot as plt

    objects = data.keys()
    y_pos = np.arange(len(objects))
    performance = data.values()

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel(y_label)
    plt.title(title)

    plt.show()


def read_corpures_dataset():
    entries = os.listdir("resources/CORPURES/Dataset-txt-format")
    files_data = []
    for entry in entries:
        f = open("resources/CORPURES/Dataset-txt-format/" + entry, encoding="utf8")
        file_data = {}
        text = ""

        is_summary = False
        summary_line_no = 10000000
        summary_text = ""
        for i, line in enumerate(f):
            if "Summary" in line:
                is_summary = True
                summary_line_no = i + 1
                continue
            else:
                if i == 1:
                    file_data["Title"] = line.strip()
                elif i >= 4 and is_summary == False:
                    text += line

            if i >= summary_line_no:
                summary_text += line

        file_data["Text"] = text
        file_data["Summary"] = summary_text
        file_data['FileName'] = entry
        category = re.findall('([a-zA-Z ]*)\d*.*', entry)
        file_data['category'] = category[0]
        files_data.append(file_data)
        f.close()
    return files_data


# Remove stop words from text
from typing import FrozenSet

# Urdu Language Stop words list
STOP_WORDS: FrozenSet[str] = frozenset("""
 آ آئی آئیں آئے آتا آتی آتے آس آمدید آنا آنسہ آنی آنے آپ آگے آہ آہا آیا اب ابھی ابے
 ارے اس اسکا اسکی اسکے اسی اسے اف افوہ البتہ الف ان اندر انکا انکی انکے انہوں انہی انہیں اوئے اور اوپر
 اوہو اپ اپنا اپنوں اپنی اپنے اپنےآپ اکثر اگر اگرچہ اہاہا ایسا ایسی ایسے ایک بائیں بار بارے بالکل باوجود باہر
 بج بجے بخیر بشرطیکہ بعد بعض بغیر بلکہ بن بنا بناؤ بند بڑی بھر بھریں بھی بہت بہتر تاکہ تاہم تب تجھ
 تجھی تجھے ترا تری تلک تم تمام تمہارا تمہاروں تمہاری تمہارے تمہیں تو تک تھا تھی تھیں تھے تیرا تیری تیرے
 جا جاؤ جائیں جائے جاتا جاتی جاتے جانی جانے جب جبکہ جدھر جس جسے جن جناب جنہوں جنہیں جو جہاں جی جیسا
 جیسوں جیسی جیسے حالانکہ حالاں حصہ حضرت خاطر خالی خواہ خوب خود دائیں درمیان دریں دو دوران دوسرا دوسروں دوسری دوں
 دکھائیں دی دیئے دیا دیتا دیتی دیتے دیر دینا دینی دینے دیکھو دیں دیے دے ذریعے رکھا رکھتا رکھتی رکھتے رکھنا رکھنی
 رکھنے رکھو رکھی رکھے رہ رہا رہتا رہتی رہتے رہنا رہنی رہنے رہو رہی رہیں رہے ساتھ سامنے ساڑھے سب سبھی
 سراسر سمیت سوا سوائے سکا سکتا سکتے سہ سہی سی سے شاید شکریہ صاحب صاحبہ صرف ضرور طرح طرف طور علاوہ عین
 فقط فلاں فی قبل قطا لئے لائی لائے لاتا لاتی لاتے لانا لانی لانے لایا لو لوجی لوگوں لگ لگا لگتا
 لگتی لگی لگیں لگے لہذا لی لیا لیتا لیتی لیتے لیکن لیں لیے لے ماسوا مت مجھ مجھی مجھے محترم محترمہ محض
 مرا مرحبا مری مرے مزید مس مسز مسٹر مطابق مل مکرمی مگر مگھر مہربانی میرا میروں میری میرے میں نا نزدیک
 نما نہ نہیں نیز نیچے نے و وار واسطے واقعی والا والوں والی والے واہ وجہ ورنہ وغیرہ ولے وگرنہ وہ وہاں
 وہی وہیں ویسا ویسے ویں پاس پایا پر پس پلیز پون پونی پونے پھر پہ پہلا پہلی پہلے پیر پیچھے چاہئے
 چاہتے چاہیئے چاہے چلا چلو چلیں چلے چناچہ چند چونکہ چکی چکیں چکے ڈالنا ڈالنی ڈالنے ڈالے کئے کا کاش کب کبھی
 کدھر کر کرتا کرتی کرتے کرم کرنا کرنے کرو کریں کرے کس کسی کسے کم کن کنہیں کو کوئی کون کونسا
 کونسے کچھ کہ کہا کہاں کہہ کہی کہیں کہے کی کیا کیسا کیسے کیونکر کیونکہ کیوں کیے کے گئی گئے گا گنا
 گو گویا گی گیا ہائیں ہائے ہاں ہر ہرچند ہرگز ہم ہمارا ہماری ہمارے ہمی ہمیں ہو ہوئی ہوئیں ہوئے ہوا
 ہوبہو ہوتا ہوتی ہوتیں ہوتے ہونا ہونگے ہونی ہونے ہوں ہی ہیلو ہیں ہے یا یات یعنی یک یہ یہاں یہی یہیں
""".split())


def remove_stopwords(text: str):
    return " ".join(word for word in text.split() if word not in STOP_WORDS)


def get_word_tokens(text_sents_tokenized):
    doc_words = []
    for sentence_tokenized in text_sents_tokenized:
        doc_words.extend(sentence_tokenized)
    return doc_words


def generate_features(sen, text_sents_pos, text_sents_tokenized, word_freq, position, doc_pos_features, category,
                      ur_cue_words, sent_text):
    '''
    Args:
        sent: array of words
    '''
    features = {}
    total_sentences = len(text_sents_pos)

    features["doc_nouns"] = doc_pos_features['doc_nouns']
    features["doc_verbs"] = doc_pos_features['doc_verbs']
    features["doc_adjcs"] = doc_pos_features['doc_adjcs']
    features["doc_advbs"] = doc_pos_features['doc_advbs']
    features["doc_nums"] = doc_pos_features['doc_nums']

    # Compute Doc unaware features:
    features["ordinal_position"] = 1 / position
    features["length_of_sentence"] = len(sen)
    Features.doc_unaware_pos_ratio(features, sen)

    # find cue words/cue phrases in sentence
    features["cue_words"] = Features.find_cue_phrases(ur_cue_words, sent_text)

    # Compute Explicit Doc feature:
    features["explicit_doc_sents"] = len(text_sents_pos)
    features["explicit_doc_words"] = find_doc_words(text_sents_tokenized)
    features["explicit_topical_categoty"] = corpures_category_mapping(category)

    # Compute Doc aware features
    features["doc_aware_relative_length"] = Features.find_relative_length(text_sents_tokenized, sen)
    features["doc_aware_tfisf"] = Features.tf_isf_score(sen, text_sents_tokenized, word_freq)
    features["doc_aware_cosine_position"] = Features.cosine_position_score(position, total_sentences)
    Features.doc_aware_pos_ratios(features, sen)
    return features


def find_doc_words(text_sents_tokenized):
    total_doc_words = 0;
    for sen in text_sents_tokenized:
        total_doc_words += len(sen)

    return total_doc_words


def find_doc_pos(text_sents_pos):
    nn_count = 0
    ve_count = 0
    aj_count = 0
    av_count = 0
    num_count = 0
    for sentence in text_sents_pos:
        for index, word_tuple in enumerate(sentence):
            pos = word_tuple[1]
            if pos == "NN" or pos == "PN":
                nn_count += 1
            elif pos == "VB" or pos == "AA" or pos == "TA":
                ve_count += 1
            elif pos == "ADJ":
                aj_count += 1
            elif pos == "ADV":
                av_count += 1
            elif pos == "CA" or pos == "OR" or pos == "FR" or pos == "MUL":
                num_count += 1

    doc_pos_features = {
        "doc_nouns": nn_count if nn_count > 0 else 1,
        "doc_verbs": ve_count if ve_count > 0 else 1,
        "doc_adjcs": aj_count if aj_count > 0 else 1,
        "doc_advbs": av_count if av_count > 0 else 1,
        "doc_nums": num_count if num_count > 0 else 1
    }
    return doc_pos_features


def sentence_word(sen):
    words = []
    for index, word_tuple in enumerate(sen):
        words.append(word_tuple[0])
    return words


def tokenized_text_to_string(tokenized_text):
    text = ""
    for sent in tokenized_text:
        text += " ".join(sent)

    return text
