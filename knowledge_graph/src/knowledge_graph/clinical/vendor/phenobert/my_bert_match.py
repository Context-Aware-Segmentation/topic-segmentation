import torch.nn
from fastNLP import Vocabulary
from string import punctuation
from fastNLP.embeddings.My_bert_embedding import BertEmbedding
from util import biobert_path

embed = BertEmbedding(Vocabulary(), model_dir_or_name=biobert_path,include_cls_sep=True, pool_method="max")

def addWordPiece(instance):
    sentence=instance["p_words"]
    results=[]
    for word in sentence:
        results.append(embed.model.tokenzier.convert_tokens_to_ids(embed.model.tokenzier.wordpiece_tokenizer.tokenize(word)))
    return results

def addWords(instance):
    sentence = instance["raw_words"].lower()
    inter = sentence.split("::")
    word1 = inter[0]
    word2 = inter[1]
    p_word1 = ""
    for c in word1:
        if c in punctuation:
            p_word1 += " "
        else:
            p_word1 += c
    p_word2 = ""
    for c in word2:
        if c in punctuation:
            p_word2 += " "
        else:
            p_word2 += c
    words = p_word1 + " [SEP] " + p_word2
    return words.split()

def addSeqlen(instance):
    words = instance["p_words"]
    return len(words)

def processNum(instance):
    inter=instance["t_words"]
    results=[len(item) for item in inter]
    return results

def processItem(instance):
    inter = instance["t_words"]
    results=[]
    for item in inter:
        results.extend(item)
    return results
