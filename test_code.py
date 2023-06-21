import os
import re

import urduhack
from urduhack import normalize
from urduhack.tokenization import sentence_tokenizer
from urduhack.tokenization import word_tokenizer
# import pandas as pd
from urduhack.preprocessing import normalize_whitespace, remove_punctuation

import utilities

"""
df = pd.read_csv("resources/CORPURES/Summary_data.csv")
gk = df.groupby('FileName')

# iterate over each group
for group_name, df_group in gk:
    print('\nCREATE TABLE {}'.format(group_name))

    for row_index, row in df_group.iterrows():
        print('\nRow index= {} , category= {}'.format(row_index, row['Category']))


"""

f = open("resources/CORPURES/remaining/Int28.txt", encoding="utf8")
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
            text += line.strip()

    if i >= summary_line_no:
        summary_text += line.strip()

text = text.replace("۔", "۔<eos>")
text = text.replace(".", "<eos>.")
text = text.replace("؟", "<eos>؟")
sent_tok = text.split("<eos>")

# sent_tok = re.split("[.؟۔]", text)
sent_tok.pop()
print(sent_tok)






