from Summ import summ
from rouge import Rouge

def best_rouge_f(score1, score2):
    sum_F1 = score1["rouge-1"]["f"] + score1["rouge-2"]["f"] + score1["rouge-l"]["f"]
    sum_F2 = score2["rouge-1"]["f"] + score2["rouge-2"]["f"] + score2["rouge-l"]["f"]
    if sum_F2 > sum_F1:
        return score2
    return score1

def evaluate_summarizer(clf, dataset, used_features, ur_cue_words, remove_stopwords=False):
    # Load urdu cue words

    empty_score = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    total_scores = {
        'rouge-1': {'p': 0, 'f': 0, 'r': 0},
        'rouge-2': {'p': 0, 'f': 0, 'r': 0},
        'rouge-l': {'p': 0, 'f': 0, 'r': 0}
    }
    avg_scores = empty_score
    total_summaries = 0
    rouge = Rouge()

    # diff_summs = 0
    for index, doc in enumerate(dataset):
        total_summaries += 1
        text = doc["Text"]
        ref = doc["Summary"]

        text_sents_tokenized = doc["Text_Sents_Tokenized"]
        text_sents_pos = doc["Text_Sents_POS"]
        text_sents = doc["Text_Sents"]

        title = doc["Title"]
        summary_sents_tokenized = doc["Summary_Sents_Tokenized"]
        summary_sents_pos = doc["Summary_Sents_POS"]
        summary_sents = doc["Summary_Sents"]

        category = doc["category"]
        best_score = empty_score

        ref_len = len(summary_sents_tokenized)

        summary = summ(text_sents_tokenized, text_sents_pos, text_sents, title, text, clf, category, used_features,
                       ur_cue_words, ref_len)

        if summary == "":
            continue
        try:
            scores = rouge.get_scores(ref, summary)[0]
        except:
            print(ref)
            print(summary)
            o = 1
            o += 1
        """f_file = open('/tmp/summaries/' + ref_key + str(scores["rouge-1"]["f"]) + '.txt', '+w')
        f_file.writelines(lines)
        f_file.close()"""
        best_score = best_rouge_f(best_score, scores)

        for test_type in best_score:
            for param in best_score[test_type]:
                total_scores[test_type][param] += best_score[test_type][param]

    total_docs = len(dataset)
    for test_type in total_scores:
        for param in total_scores[test_type]:
            avg_scores[test_type][param] = total_scores[test_type][param] / total_summaries
    return avg_scores




