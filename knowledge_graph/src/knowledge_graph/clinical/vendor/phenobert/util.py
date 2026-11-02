import torch
from torch.utils.data import Dataset, DataLoader
import json
import nltk
import re
import os
import numpy as np
import unicodedata
from nltk.tokenize import PunktSentenceTokenizer, TreebankWordTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from knowledge_graph.paths import REPOSITORY

# file path definition: PhenoBERT data under data/phenobert, released weights under checkpoints/phenobert
data_path = str(REPOSITORY / "data" / "phenobert") + "/"
weights_path = str(REPOSITORY / "checkpoints" / "phenobert") + "/"
hpo_json_path = data_path + "hpo.json"
fasttext_model_path = weights_path + "embeddings/fasttext_pubmed.bin"
stopwords_file_path = data_path + "stopwords.txt"
num2word_file_path = data_path + "NUM.txt"
biobert_path = weights_path + "embeddings/biobert_v1.1_pubmed"
cnn_model_path = weights_path + "models/HPOModel_H/model_layer1.pkl"
sub_model_path = weights_path + "models/HPOModel_H/model_l1_{}.pkl"
bert_model_path = weights_path + "models/bert_model_max_triple.pkl"
device = torch.device("cuda:0" if torch.cuda.is_available() else torch.device("cpu"))
wnl = WordNetLemmatizer()

class HPO_class:

    def __init__(self, dic):
        self.id = dic["Id"]
        self.name = dic["Name"]
        self.alt_id = dic["Alt_id"]
        self.defination = dic["Def"]
        self.comment = dic["Comment"]
        self.synonym = dic["Synonym"]
        self.xref = dic["Xref"]
        self.is_a = dic["Is_a"]
        self.father = set(dic["Father"].keys())
        self.child = set(dic["Child"].keys())
        self.son = set(dic["Son"].keys())


class PhraseDataSet4predict(Dataset):

    def __init__(self, phrase_list, fasttext_model):
        self.data = []
        self.phrase_list = []
        self.max_seq_len = 30
        self.fasttext_model = fasttext_model
        self.embedding_dim = self.fasttext_model.get_dimension()

        for phrase_item in phrase_list:
            if isinstance(phrase_item, PhraseItem):
                p_phrase = processStr(phrase_item.toString())
            elif isinstance(phrase_item, str):
                p_phrase = processStr(phrase_item)
            else:
                p_phrase = ""
            if len(p_phrase) > 0:
                self.phrase_list.append(phrase_item)
                self.data.append(p_phrase)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        p_phrase = self.data[idx]
        # [len(phrase), embedding_dim]
        data = np.concatenate([self.fasttext_model.get_word_vector(word).reshape(1, -1) for word in p_phrase])
        pad_data = np.zeros(shape=(self.max_seq_len, data.shape[1]))
        pad_data[:data.shape[0]] = data
        seq_len = data.shape[0]
        sample = {"data": pad_data, "seq_len": seq_len}
        return sample


def PhraseDataSet4predictFunc(phrase_item, fasttext_model, max_seq_len=30):
    if isinstance(phrase_item, PhraseItem):
        p_phrase = processStr(phrase_item.toString())
    elif isinstance(phrase_item, str):
        p_phrase = processStr(phrase_item)
    else:
        p_phrase = ""
    if len(p_phrase) == 0:
        return None
    data = np.concatenate([fasttext_model.get_word_vector(word).reshape(1, -1) for word in p_phrase])
    pad_data = np.zeros(shape=(max_seq_len, data.shape[1]))
    pad_data[:data.shape[0]] = data
    return [torch.tensor(pad_data).unsqueeze(0).float().to(device), None]


class WordItem:
    lemma_dict = {}
    def __init__(self, text, start, end):
        self.text = text.lower()
        self.start = start
        self.end = end

def getNum2Word(file_path):
    Num2Word = {}
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            inter = line.strip().split("\t")
            Num2Word[inter[0]] = inter[1]
    return Num2Word

class PhraseItem:

    Num2Word = getNum2Word(num2word_file_path)
    StopWords = stopwords.words("english")

    def __init__(self, word_items):
        self.word_items = word_items
        self.simple_items = []
        self.simplify()
        self.locs_set = set([i.start for i in word_items])
        self.start_loc = self.word_items[0].start
        self.end_loc = self.word_items[-1].end
        self.no_flag = False


    def simplify(self):
        for word_item in self.word_items:
            if word_item.text in PhraseItem.Num2Word:
                self.simple_items.append(WordItem(PhraseItem.Num2Word[word_item.text], word_item.start, word_item.end))
            elif word_item.text in PhraseItem.StopWords or isNum(word_item.text):
                continue
            else:
                self.simple_items.append(word_item)


    def toString(self):
        return " ".join([i.text for i in self.word_items])

    def toSimpleString(self):
        return " ".join([i.text for i in self.simple_items])

    def include(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set) or self.locs_set.issuperset(phrase_item.locs_set):
            return True
        return False

    def issubset(self, phrase_item):
        if self.locs_set.issubset(phrase_item.locs_set):
            return True
        return False

    def set_no_flag(self):
        self.no_flag = True

    def __len__(self):
        return len(self.word_items)


class HPOTree:

    def __init__(self):
        with open(hpo_json_path, encoding="utf-8") as json_file:
            self.data = json.loads(json_file.read())

        # tmp_list = [[HPO_class(self.data[i]).name, len(HPO_class(self.data[i]).child)] for i in
        #             HPO_class(self.data["HP:0000118"]).son]
        # tmp_list = sorted(tmp_list, key=lambda x: x[1], reverse=True)
        # print(tmp_list)

        # print(HPO_class(self.data["HP:0000001"]).father)
        # print("HP:0000001" in HPO_class(self.data["HP:0000001"]).child)

        self.root = "HP:0000118"
        self.phenotypic_abnormality = HPO_class(self.data[self.root]).child
        self.phenotypic_abnormalityNT = set(list(self.phenotypic_abnormality))
        self.phenotypic_abnormality.add(self.root)
        self.hpo_list = sorted(list(self.phenotypic_abnormality))
        self.n_concept = len(self.hpo_list)
        self.hpo2idx = {hpo: idx for idx, hpo in enumerate(self.hpo_list)}
        # Add None
        self.hpo2idx["None"] = len(self.hpo_list)
        self.idx2hpo = {self.hpo2idx[hpo]: hpo for hpo in self.hpo2idx}
        self.alt_id_dict = {}
        self.p_phrase2HPO = {}
        self.depth = 0
        self.layer1 = sorted(list(HPO_class(self.data[self.root]).son))
        self.layer1_set = set(self.layer1)
        self.n_concept_l1 = len(self.layer1)
        self.hpo2idx_l1 = {hpo: idx for idx, hpo in enumerate(self.layer1)}
        # Add None
        self.hpo2idx_l1["None"] = len(self.layer1)
        self.idx2hpo_l1 = {self.hpo2idx_l1[hpo]: hpo for hpo in self.hpo2idx_l1}

        for hpo_name in self.data:
            struct = HPO_class(self.data[hpo_name])
            alt_ids = struct.alt_id
            for sub_alt_id in alt_ids:
                self.alt_id_dict[sub_alt_id] = hpo_name
            phrases = getNames(struct)
            for phrase in phrases:
                key = " ".join(sorted(processStr(phrase)))
                self.p_phrase2HPO[key] = hpo_name

    def buildHPOTree(self):

        self.depth_dict = {}
        queue = {self.root}
        visited = {self.root}
        depth = 0
        while len(queue) > 0:
            tmp = set()
            for node in queue:
                self.depth_dict[node] = depth
                son = HPO_class(self.data[node]).son
                for sub_node in son:
                    if sub_node not in visited:
                        visited.add(sub_node)
                        tmp.add(sub_node)
            queue = tmp
            depth += 1
        self.depth = depth - 1

    def getNameByHPO(self, hpo_num):
        return HPO_class(self.data[hpo_num]).name[0].lower()


    def matchPhrase2HPO(self, phrase):
        p_phrase = " ".join(sorted(processStr(phrase)))
        p_l_phrase = " ".join([WordItem.lemma_dict[i] if i in WordItem.lemma_dict else i for i in p_phrase.split()])
        if p_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_phrase]
        elif p_l_phrase in self.p_phrase2HPO:
            return self.p_phrase2HPO[p_l_phrase]
        return ""


    def getHPO2idx_l1(self, hpo_num):
        return self.hpo2idx_l1[hpo_num]


    def getIdx2HPO_l1(self, idx):
        return self.idx2hpo_l1[idx]

    def getMaterial4L1(self, root_l1):
        root_idx = self.getHPO2idx_l1(root_l1)
        hpo_list = HPO_class(self.data[root_l1]).child
        hpo_list.add(root_l1)
        hpo_list = sorted(hpo_list)
        n_concept = len(hpo_list)
        hpo2idx = {hpo: idx for idx, hpo in enumerate(hpo_list)}
        # Add None
        hpo2idx["None"] = len(hpo_list)
        idx2hpo = {hpo2idx[hpo]: hpo for hpo in hpo2idx}
        return root_idx, hpo_list, n_concept, hpo2idx, idx2hpo


        # # Info-based score
        # pc1=(len(struct1.child)+1)/len(self.phenotypic_abnormality)
        # pc2=(len(struct2.child)+1)/len(self.phenotypic_abnormality)
        # pc3=(len(struct3.child)+1)/len(self.phenotypic_abnormality)
        # ib_score = 2*math.log2(pc3)/(math.log2(pc1)+math.log2(pc2))
        # return ib_score


class SpanTokenizer:

    def __init__(self):
        self.tokenizer_big = PunktSentenceTokenizer()
        self.tokenizer_small = TreebankWordTokenizer()

    def tokenize(self, text):
        result = []
        sentences_span = self.tokenizer_big.span_tokenize(text)
        for start, end in sentences_span:
            sentence = text[start:end]
            tokens_span = self.tokenizer_small.span_tokenize(sentence)
            for token_start, token_end in tokens_span:
                result.append([start + token_start, start + token_end])
        return result


class ModelLoader:

    def __init__(self):
        self.models = {}


    def load_all(self, model_save_path):
        path = os.path.abspath(model_save_path)
        if path not in self.models:
            location = None if torch.cuda.is_available() else torch.device('cpu')
            self.models[path] = torch.load(path, map_location=location, weights_only=False)
        return self.models[path]


def getNames(struct):
    names = struct.name
    synonyms = struct.synonym
    # all name + synonym
    names.extend(synonyms)
    names = list(set(names))
    return names


def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def processStr(string):
    string = re.sub("(?<=[A-Z])-(?=[\d])", "", string)
    string = strip_accents(string.lower())
    string = re.sub("[-_\"\'\\\\\t\r\n‘’]", " ", string)
    all_text = string.strip().split()
    return all_text


def isNum(strings):
    try:
        float(strings)
        return True
    except ValueError:
        return False


def getStopWords():
    stopwords = set()
    with open(stopwords_file_path, encoding="utf-8") as file:
        for line in file:
            stopwords.add(line.strip())
    return stopwords


def getSpliters():
    spliters = set([word for word, pos in nltk.pos_tag(stopwords.words('english')) if pos in {'CC', 'WP', 'TO', 'WDT'}]+[',', '.', ':', ';', '(', ')', '[', ']', '/'])
    return spliters


def produceCandidateTriple(Candidate_hpos_sub_total, model, threshold):
    from fastNLP.core.utils import _move_dict_value_to_device
    from fastNLP.core.utils import _get_model_device
    from fastNLP import DataSet
    from fastNLP import DataSetIter
    from my_bert_match import addWordPiece, addSeqlen, addWords, processItem, processNum
    p_Candidate_phrases = []
    phrase_nums_per_hpo = []
    Candidate_hpos = []
    for raw_phrase, Candidate_phrase, Candidate_hpos_sub in Candidate_hpos_sub_total:
        p_Candidate_phrases.extend([raw_phrase + "::" + item for item in Candidate_phrase])
        phrase_nums_per_hpo.append(len(Candidate_phrase))
        Candidate_hpos.append(Candidate_hpos_sub)
    Candidate_dataset = DataSet({"raw_words": p_Candidate_phrases})
    Candidate_dataset.apply(addWords, new_field_name="p_words")
    Candidate_dataset.apply(addWordPiece, new_field_name="t_words")
    Candidate_dataset.apply(processItem, new_field_name="word_pieces")
    Candidate_dataset.apply(processNum, new_field_name="word_nums")
    Candidate_dataset.apply(addSeqlen, new_field_name="seq_len")
    Candidate_dataset.field_arrays["word_pieces"].is_input = True
    Candidate_dataset.field_arrays["seq_len"].is_input = True
    Candidate_dataset.field_arrays["word_nums"].is_input = True
    test_batch = DataSetIter(batch_size=128, dataset=Candidate_dataset, sampler=None)

    outputs = []
    for batch_x, batch_y in test_batch:
        _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
        outputs.append(model.forward(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])['pred'])
    outputs = torch.cat(outputs)
    outputs = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()
    # print(outputs.size)
    results_2 = np.array([item[2] for item in outputs])
    results_1 = np.array([item[1] for item in outputs])

    count = 0
    index = 0
    ans = []
    for group_num in phrase_nums_per_hpo:
        g_results_2 = results_2[index:index+group_num]
        g_results_1 = results_1[index:index+group_num]
        Candidate_hpos_sub = Candidate_hpos[count]
        index += group_num
        count += 1
        if max(g_results_2) >= threshold:
            ans.append([Candidate_hpos_sub[int(np.argmax(g_results_2))], max(g_results_2), "2"])
            continue
        if max(g_results_1) >= threshold:
            ans.append([Candidate_hpos_sub[int(np.argmax(g_results_1))], max(g_results_1), "1"])
            continue
        ans.append(["None", None, "0"])
    return ans

def produceCandidateTripleSlow(raw_phrase, Candidate_phrases, model, Candidate_hpos_sub, threshold):
    from fastNLP.core.utils import _move_dict_value_to_device
    from fastNLP.core.utils import _get_model_device
    from fastNLP import DataSet
    from fastNLP import DataSetIter
    from my_bert_match import addWordPiece, addSeqlen, addWords, processItem, processNum
    p_Candidate_phrases = [raw_phrase + "::" + item for item in Candidate_phrases]
    Candidate_dataset = DataSet({"raw_words": p_Candidate_phrases})
    Candidate_dataset.apply(addWords, new_field_name="p_words")
    Candidate_dataset.apply(addWordPiece, new_field_name="t_words")
    Candidate_dataset.apply(processItem, new_field_name="word_pieces")
    Candidate_dataset.apply(processNum, new_field_name="word_nums")
    Candidate_dataset.apply(addSeqlen, new_field_name="seq_len")
    Candidate_dataset.field_arrays["word_pieces"].is_input = True
    Candidate_dataset.field_arrays["seq_len"].is_input = True
    Candidate_dataset.field_arrays["word_nums"].is_input = True
    test_batch = DataSetIter(batch_size=10, dataset=Candidate_dataset, sampler=None)

    outputs = []
    for batch_x, batch_y in test_batch:
        _move_dict_value_to_device(batch_x, batch_y, device=_get_model_device(model))
        outputs.append(model.forward(batch_x["word_pieces"], batch_x["word_nums"], batch_x["seq_len"])['pred'])
    outputs = torch.cat(outputs)
    outputs = torch.nn.functional.softmax(outputs, dim=1).cpu().detach().numpy()

    results_2 = np.array([item[2] for item in outputs])
    results_1 = np.array([item[1] for item in outputs])

    if max(results_2) >= threshold:
        return Candidate_hpos_sub[int(np.argmax(results_2))], max(results_2), "2"

    if max(results_1) >= threshold:
        return Candidate_hpos_sub[int(np.argmax(results_1))], max(results_1), "1"

    return "None", None, "0"

def process_text2phrases(text, clinical_ner_model):
    tokenizer = SpanTokenizer()
    spliters = getSpliters()
    stopwords = getStopWords()
    text = strip_accents(text.lower())
    text = re.sub("[-_\"\'\\\\\t‘’]", " ", text)
    text = re.sub("(?<=[\w])[\r\n]", ".", text)

    clinical_docs = clinical_ner_model(text)
    sub_sentences = []

    for sent_c in clinical_docs.sentences:
        clinical_tokens = sent_c.tokens

        # Stanza
        flag = False
        curSentence = []
        tmp = set()
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            if "PROBLEM" in clinical_tokens[i].ner and wi.text not in {',', '.', ':', ';', '(', ')', '[', ']'}:
                curSentence.append(wi)
            else:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    sub_sentences.append(phrase_item)
                    tmp.update(phrase_item.locs_set)
                    flag = True
                curSentence = []

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            sub_sentences.append(phrase_item)
            tmp.update(phrase_item.locs_set)
            flag = True

        # phrase segmentation
        if not flag:
            continue
        curSentence = []
        for i in range(len(clinical_tokens)):
            wi = WordItem(clinical_tokens[i].text, clinical_tokens[i].start_char, clinical_tokens[i].end_char)
            text_lemma = wnl.lemmatize(clinical_tokens[i].text)
            if clinical_tokens[i].text not in WordItem.lemma_dict:
                WordItem.lemma_dict[clinical_tokens[i].text] = text_lemma
            if clinical_tokens[i].text in spliters:
                if len(curSentence) > 0:
                    phrase_item = PhraseItem(curSentence)
                    if len(phrase_item.locs_set & tmp) == 0:
                        sub_sentences.append(phrase_item)
                curSentence = []
            else:
                curSentence.append(wi)

        if len(curSentence) > 0:
            phrase_item = PhraseItem(curSentence)
            if len(phrase_item.locs_set & tmp) == 0:
                sub_sentences.append(phrase_item)

    for phrase_item in sub_sentences:
        flag = False
        for token in phrase_item.word_items:
            if token.text.lower() in {"no", "not", "none", "negative", "non", "never", "few", "lower", "fewer", "less",
                                      "normal"}:
                flag = True
                break
        if flag:
            phrase_item.set_no_flag()

    sub_sentences_ = []
    for idx, pi in enumerate(sub_sentences):
        sub_locs = [[i + pi.start_loc, j + pi.start_loc] for i, j in
                    tokenizer.tokenize(text[pi.start_loc:pi.end_loc])]
        sub_phrases = []
        curr_phrase = []
        for loc in sub_locs:
            wi = WordItem(text[loc[0]:loc[1]], loc[0], loc[1])
            if wi.text in {"and", "or", "/"}:
                if len(curr_phrase) > 0:
                    sub_phrases.append(PhraseItem(curr_phrase))
                    sub_phrases[-1].no_flag = pi.no_flag
                curr_phrase = []
            else:
                curr_phrase.append(wi)
        if len(curr_phrase) > 0:
            sub_phrases.append(PhraseItem(curr_phrase))
            sub_phrases[-1].no_flag = pi.no_flag

        for item in sub_phrases:
            sub_sentences_.append(item)

        if len(sub_phrases) == 2:
            if len(sub_phrases[0]) >= 1 and len(sub_phrases[1]) == 1:
                tmp = sub_phrases[0].word_items[:-1][:]
                tmp.extend(sub_phrases[1].word_items)
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag
            elif len(sub_phrases[0]) == 1 and len(sub_phrases[1]) >= 1:
                tmp = sub_phrases[0].word_items[:]
                tmp.extend(sub_phrases[1].word_items[1:])
                sub_sentences_.append(PhraseItem(tmp))
                sub_sentences_[-1].no_flag = pi.no_flag

    sub_sentences = sub_sentences_

    # print([i.toSimpleString() for i in sub_sentences])

    phrases_list = []
    for pi in sub_sentences:
        tmp = pi.toSimpleString()
        if isNum(tmp) or len(tmp) <= 1:
            continue
        for i in range(len(pi.simple_items)):
            for j in range(10):
                if i + j == len(pi.simple_items):
                    break
                if len(pi.simple_items[i:i + j + 1]) == 1:
                    tmp_str = pi.simple_items[i:i + j + 1][0].text
                    if tmp_str in stopwords or isNum(tmp_str):
                        continue
                phrases_list.append(PhraseItem(pi.simple_items[i:i + j + 1]))
                phrases_list[-1].no_flag = pi.no_flag

    # print(len(phrases_list))
    # print([i.toString() for i in phrases_list])
    return phrases_list


def annotate_phrases(text, phrases_list, hpo_tree, fasttext_model, cnn_model, bert_model,
                     output_file_path, device, param1, param2, param3, use_longest, use_step_3, loader=None):

    if loader is None:
        loader = ModelLoader()
    result_list = []
    if output_file_path is not None:
        output_file = open(output_file_path, "w", encoding="utf-8")
    else:
        output_file = ""

    next_phrases_list = []
    for phrase_item in phrases_list:
        raw_phrase = phrase_item.toString()

        d_match = hpo_tree.matchPhrase2HPO(raw_phrase)
        if d_match != "" and d_match in hpo_tree.phenotypic_abnormalityNT:
            # print(phrase_item.toString(), d_match)
            result_list.append([phrase_item, d_match, 1.0])
        else:
            flag = True
            for known_phrase_item, _, _ in result_list:
                if phrase_item.issubset(known_phrase_item):
                    flag = False
                    break
            if flag:
                next_phrases_list.append(phrase_item)

    # print([i.toString() for i in next_phrases_list])

    dataset = PhraseDataSet4predict(next_phrases_list, fasttext_model)
    dataloader = DataLoader(dataset, batch_size=256, shuffle=False)

    total_hpo_list = []
    total_hpo2idx = []
    total_idx2hpo = []
    total_model = []
    total_l1_root = hpo_tree.layer1
    for sub_l1_root in total_l1_root:
        # get idx2hpo hpo2idx for given HPO
        root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = hpo_tree.getMaterial4L1(sub_l1_root)
        total_hpo_list.append(hpo_list)
        total_hpo2idx.append(hpo2idx)
        total_idx2hpo.append(idx2hpo)
        sub_model_save_path = sub_model_path.format(root_idx)
        sub_model = loader.load_all(sub_model_save_path)
        sub_model.eval()
        total_model.append(sub_model)

    samples = 0

    with torch.no_grad():
        cnn_model.eval()
        bert_model.eval()

        for data in dataloader:
            # CNN input
            input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
            batch_num = input_data[0].size(0)
            y = cnn_model(input_data)
            phrase_items = dataset.phrase_list[samples:samples + batch_num]
            samples += batch_num


            if not use_step_3:

                prediction = y.argsort().tolist()
                scores_p = y.sort()[0] >= param1
                Candidate_hpos = [
                    set([hpo_tree.getIdx2HPO_l1(prediction[idx1][idx2]) for idx2 in range(len(prediction[idx1])) if
                         scores_p[idx1][idx2]]) for idx1 in range(len(prediction))]
                for i, j in zip(phrase_items, Candidate_hpos):
                    if len(j) > 0 and "None" not in j:
                        Candidate_hpos_sub = []
                        for l1_hpo in j:
                            l1_idx = hpo_tree.getHPO2idx_l1(l1_hpo)
                            y_sub = total_model[l1_idx](PhraseDataSet4predictFunc(i, fasttext_model)).squeeze()
                            prediction_sub = y_sub.topk(5)[1].tolist()
                            scores_p_sub = torch.softmax(y_sub, dim=0).topk(5)[0].tolist()
                            Candidate_hpos_sub.extend(
                                [[total_idx2hpo[l1_idx][prediction_sub[idx]], scores_p_sub[idx]] for idx in
                                 range(len(prediction_sub)) if scores_p_sub[idx] >= param2])
                        Candidate_hpos_sub.sort(key=lambda x: x[1], reverse=True)
                        if len(Candidate_hpos_sub) != 0 and Candidate_hpos_sub[0][0] != "None":
                            result_list.append([i, Candidate_hpos_sub[0][0], Candidate_hpos_sub[0][1]])
            else:
                prediction = y.argsort().tolist()
                scores_p = y.sort()[0]
                Candidate_hpos = [
                    set([hpo_tree.getIdx2HPO_l1(prediction[idx1][idx2]) for idx2 in range(len(prediction[idx1])) if
                         scores_p[idx1][idx2] >= param1]) for idx1 in range(len(prediction))]

                Candidate_hpos_sub_total = []
                next_phrase_items = []
                for phrase_item, j in zip(phrase_items, Candidate_hpos):
                    # print(phrase_item.toString(), j)
                    if len(j) > 0 and "None" not in j:
                        Candidate_hpos_sub = set()
                        for l1_hpo in j:
                            l1_idx = hpo_tree.getHPO2idx_l1(l1_hpo)
                            y_sub = total_model[l1_idx](
                                PhraseDataSet4predictFunc(phrase_item, fasttext_model)).squeeze()
                            if y_sub.size(0) > 10:
                                prediction_sub = y_sub.topk(10)[1].tolist()
                                scores_p_sub = torch.softmax(y_sub, dim=0).topk(10)[0].tolist()
                            else:
                                prediction_sub = y_sub.topk(y_sub.size(0))[1].tolist()
                                scores_p_sub = torch.softmax(y_sub, dim=0).topk(y_sub.size(0))[0].tolist()
                            Candidate_hpos_sub.update(
                                [total_idx2hpo[l1_idx][prediction_sub[idx]] for idx in range(len(prediction_sub)) if
                                 scores_p_sub[idx] >= param2])

                            # for idx in range(len(prediction_sub)):
                            #     print(phrase_item.toString(), total_idx2hpo[l1_idx][prediction_sub[idx]], scores_p_sub[idx])

                        # print(phrase_item.toString(), Candidate_hpos_sub)

                        if len(Candidate_hpos_sub) != 0 and "None" not in Candidate_hpos_sub:
                            Candidate_hpos_sub = list(Candidate_hpos_sub)
                            candidate_phrase = [hpo_tree.getNameByHPO(item) for item in Candidate_hpos_sub]
                            raw_phrase = phrase_item.toString()
                            # print(raw_phrase, candidate_phrase)
                            Candidate_hpos_sub_total.append([raw_phrase, candidate_phrase, Candidate_hpos_sub])
                            next_phrase_items.append(phrase_item)
                if len(Candidate_hpos_sub_total) > 0:
                    ans = produceCandidateTriple(Candidate_hpos_sub_total, bert_model, param3)
                    for sub_ans, phrase_item in zip(ans, next_phrase_items):
                        if sub_ans[0] != "None":
                            result_list.append([phrase_item, sub_ans[0], sub_ans[1]])

        idx_to_remove = set()

        for idx1 in range(len(result_list)):
            if idx1 in idx_to_remove:
                continue
            for idx2 in range(idx1 + 1, len(result_list)):
                if idx2 in idx_to_remove:
                    continue
                if result_list[idx1][0].include(result_list[idx2][0]):
                    if use_longest:
                        if len(result_list[idx1][0]) > len(result_list[idx2][0]):
                            idx_to_remove.add(idx2)
                        else:
                            idx_to_remove.add(idx1)
                    else:
                        if result_list[idx1][1] == result_list[idx2][1]:
                            if len(result_list[idx1][0]) > len(result_list[idx2][0]):
                                idx_to_remove.add(idx1)
                            else:
                                idx_to_remove.add(idx2)

        result_list = sorted([result_list[idx] for idx in range(len(result_list)) if idx not in idx_to_remove],
                             key=lambda x: x[0].start_loc)
		
        for item in result_list:
            if not item[0].no_flag:
                if output_file_path is not None:
                    output_file.write(
                        f"{item[0].start_loc}\t{item[0].end_loc}\t{text[item[0].start_loc:item[0].end_loc]}\t{item[1]}\t{'%.2f' % (item[2])}\n")
                else:
                    output_file += f"{item[0].start_loc}\t{item[0].end_loc}\t{text[item[0].start_loc:item[0].end_loc]}\t{item[1]}\t{'%.2f' % (item[2])}\n"
            else:
                if output_file_path is not None:
                    output_file.write(
                        f"{item[0].start_loc}\t{item[0].end_loc}\t{text[item[0].start_loc:item[0].end_loc]}\t{item[1]}\t{'%.2f' % (item[2])}\tNeg\n")
                else:
                    output_file += f"{item[0].start_loc}\t{item[0].end_loc}\t{text[item[0].start_loc:item[0].end_loc]}\t{item[1]}\t{'%.2f' % (item[2])}\tNeg\n"

        if output_file_path is not None:
            output_file.close()
            return None
        else:
            return output_file