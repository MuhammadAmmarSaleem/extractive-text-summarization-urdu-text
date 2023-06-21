import spacy
import urduhack

"""
nlp = spacy.blank('ur')

doc = nlp("کچھ ممالک ایسے بھی ہیں جہاں اس برس روزے کا دورانیہ 20 گھنٹے تک ہے۔")

print("Urdu Tokenization using SpaCy")

for word in doc:
    print(word.text)
"""

load_list = urduhack.utils.pickle_load("corpures_preprocessed")
for item in load_list:
    text_sents_tokenized = item["Text_Sents_Tokenized"]
    text_sents_pos = item["Text_Sents_POS"]