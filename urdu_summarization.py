import os
import urduhack
from nltk.probability import FreqDist
from rouge import Rouge
from urduhack import normalize
from urduhack.tokenization import sentence_tokenizer
from urduhack.tokenization import word_tokenizer

import Features
from utilities import *


def remove_stop_words(words):
    # This should be read once instead of every time this function is caled
    return [word for word in words if word not in ur_stop_words]


def are_similar_rouge(sen1, sen2):
    scores = rouge.get_scores(sen1, sen2)
    return (scores[0]['rouge-2']['f'] >= 0.7)


def are_similar(sen1, sen2):
    threshold = similarity_threshold
    denominator = float(len(set(sen1).union(sen2)))
    if denominator > 0:
        ratio = len(set(sen1).intersection(sen2)) / denominator
    else:
        ratio = 0
    return ratio >= threshold, ratio


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


def read_urdu_stop_words():
    f = open("resources/urdu_stopwords/stopwords-ur.txt", encoding="utf8")
    ur_stop_words = []
    for i, line in enumerate(f):
        ur_stop_words.append(line.strip())

    return ur_stop_words


def build_feature_set_new():
    dataset = urduhack.utils.pickle_load("corpures_preprocessed_train")
    id = 0
    output = {}
    output2 = {}

    for index, doc in enumerate(dataset):
        feature_set = []
        text_sents_tokenized = doc["Text_Sents_Tokenized"]
        text_sents_pos = doc["Text_Sents_POS"]
        text_sents = doc["Text_Sents"]

        title = doc["Title"]
        category = doc["category"]
        summary_sents_tokenized = doc["Summary_Sents_Tokenized"]
        summary_sents_pos = doc["Summary_Sents_POS"]

        all_words = get_word_tokens(text_sents_tokenized)
        word_freq = FreqDist(all_words)

        position = 0

        doc_pos_features = find_doc_pos(text_sents_pos)
        for sen_index, sen in enumerate(text_sents_pos):
            sen_words = sentence_word(sen)
            id += 1
            position += 1
            similar = False
            i = 0

            sent_text = text_sents[sen_index]
            features = generate_features(sen, text_sents_pos, text_sents_tokenized, word_freq, position,
                                         doc_pos_features, category, ur_cue_words, sent_text)

            while not similar and i < len(summary_sents_tokenized):
                sen2 = summary_sents_tokenized[i]
                (similar, similarity) = are_similar(sen_words, sen2)
                features['target'] = similarity
                features['id'] = id
                output2[id] = " ".join(sen_words)
                i += 1
            feature_set.append((features, similar))

        output[index] = feature_set
    return output, output2


'''normalizer = Normalizer()
normalizer.normalize('اصلاح نويسه ها و استفاده از نیم‌فاصله پردازش را آسان مي كند')
'اصلاح نویسه‌ها و استفاده از نیم‌فاصله پردازش را آسان می‌کند'

sent_tokenize('ما هم برای وصل کردن آمدیم! ولی برای پردازش، جدا بهتر نیست؟')
['ما هم برای وصل کردن آمدیم!', 'ولی برای پردازش، جدا بهتر نیست؟']
word_tokenize('ولی برای پردازش، جدا بهتر نیست؟')
['ولی', 'برای', 'پردازش', '،', 'جدا', 'بهتر', 'نیست', '؟']

stemmer = Stemmer()
stemmer.stem('کتاب‌ها')
'کتاب'
lemmatizer = Lemmatizer()
lemmatizer.lemmatize('می‌روم')
'رفت#رو'

tagger = POSTagger(model='resources/postagger.model')
tafs = tagger.tag(word_tokenize('ما بسیار کتاب می‌خوانیم'))
[('ما', 'PRO'), ('بسیار', 'ADV'), ('کتاب', 'N'), ('می‌خوانیم', 'V')]
print(tafs)
'''

ur_cue_words = read_file("resources/cuewords_ur").split("\n")[:-1]
ur_stop_words = read_urdu_stop_words()
rouge = Rouge()

# farsnet = importEFromPaj("resources/farsnet/synset_related_to.paj")

# tagger = POSTagger(model='resources/postagger.model')
# normalizer = Normalizer()

(feats, refs) = build_feature_set_new()

f_file = open('features.json', '+w')
json.dump(feats, f_file, ensure_ascii=False)
f_file.close()

f_file = open('referense_sens.json', '+w')
json.dump(refs, f_file, ensure_ascii=False)
f_file.close()

cols = []
for key in feats:
    for (sen, target) in feats[key]:
        for attr in sen:
            cols.append(attr)
        break
    break
output = [','.join(cols) + " \r\n"]

for key in feats:
    for (sen, target) in feats[key]:
        row = []
        for attr in sen:
            row.append(str(sen[attr]))
        output.append(','.join(row) + "\r\n")
        '''str(sen['id']) + "," + str(sen['pos_nn_ratio']) + "," + str(sen['pos_ve_ratio']) + "," +\
        str(sen['pos_aj_ratio']) + "," + str(sen['pos_av_ratio']) + "," + str(sen['tfisf']) + "," + \
        str(sen['tf']) + "," + str(sen['cue_words']) + "," + str(sen['cosine_position']) + "," + \
        str(target)+ "\r\n" )'''

f_file = open('dataset.csv', '+w')
f_file.writelines(output)
f_file.close()
