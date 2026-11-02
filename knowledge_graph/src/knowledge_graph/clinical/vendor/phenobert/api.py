import os
import re
import torch
import stanza
import fasttext
import warnings
from util import process_text2phrases, annotate_phrases, ModelLoader, HPOTree, PhraseDataSet4predict, PhraseDataSet4predictFunc, produceCandidateTripleSlow
from util import cnn_model_path, bert_model_path, fasttext_model_path, sub_model_path
from model import device
from torch.utils.data import DataLoader


class PhenoBERTAPI:
    """PhenoBERT inference with instance-owned models and classifier cache."""
    def __init__(self):
        warnings.simplefilter("ignore", torch.serialization.SourceChangeWarning)
        os.environ['MKL_NUMTHREADS'] = "10"
        os.environ['OMP_NUMTHREADS'] = "10"
        self.clinical_ner_model = stanza.Pipeline('en', package='mimic', processors={'ner': 'i2b2'}, verbose=False)
        self.loader = ModelLoader()
        self.hpo_tree = HPOTree()
        self.hpo_tree.buildHPOTree()
        self.cnn_model = self.loader.load_all(cnn_model_path)
        self.bert_model = self.loader.load_all(bert_model_path)
        self.fasttext_model = fasttext.load_model(fasttext_model_path)
    def annotate_text(self, text, output=None, param1=0.8, param2=0.6, param3=0.9, use_longest=True, use_step_3=True):
        """Return tab-separated annotations, or write them to output."""
        text = re.sub("(?<=[A-Z])-(?=[\d])", "", text)
        phrases_list = process_text2phrases(text, self.clinical_ner_model)
        result = annotate_phrases(text, phrases_list, self.hpo_tree, self.fasttext_model, self.cnn_model, self.bert_model,
                                  output, device, param1, param2, param3, use_longest,
                                  use_step_3, loader=self.loader)
        return result

    def get_L1_HPO_term(self, phrases_list, param1=0.8):
        """Assign phrases to first-level HPO categories."""
        dataset = PhraseDataSet4predict(phrases_list, self.fasttext_model)
        dataloader = DataLoader(dataset, batch_size=20, shuffle=False)
        samples = 0
        res = []
        with torch.no_grad():
            self.cnn_model.eval()
            for data in dataloader:
                input_data = [data["data"].float().to(device), data["seq_len"].int().to(device)]
                batch_num = input_data[0].size(0)
                y = self.cnn_model(input_data)
                phrase_items = dataset.phrase_list[samples:samples + batch_num]
                samples += batch_num
                prediction = y.argsort().tolist()
                scores_p = y.sort()[0]
                Candidate_hpos = [
                    set([self.hpo_tree.getIdx2HPO_l1(prediction[idx1][idx2]) for idx2 in range(len(prediction[idx1])) if
                         scores_p[idx1][idx2] >= param1]) for idx1 in range(len(prediction))]
                for phrase_item, j in zip(phrase_items, Candidate_hpos):
                    res.append([phrase_item, j])
        return res

    def get_most_related_HPO_term(self, phrases_list, param1=0.8, param2=0.6, param3=0.9):
        first_step = self.get_L1_HPO_term(phrases_list, param1)
        res = []
        total_hpo_list = []
        total_hpo2idx = []
        total_idx2hpo = []
        total_model = []
        total_l1_root = self.hpo_tree.layer1
        for sub_l1_root in total_l1_root:
            # get idx2hpo hpo2idx for given HPO
            root_idx, hpo_list, n_concept, hpo2idx, idx2hpo = self.hpo_tree.getMaterial4L1(sub_l1_root)
            total_hpo_list.append(hpo_list)
            total_hpo2idx.append(hpo2idx)
            total_idx2hpo.append(idx2hpo)
            sub_model_save_path = sub_model_path.format(root_idx)
            sub_model = self.loader.load_all(sub_model_save_path)
            sub_model.eval()
            total_model.append(sub_model)
        for i, j in first_step:
            flag = True
            if len(j) > 0 and "None" not in j:
                Candidate_hpos_sub = set()
                for l1_hpo in j:
                    l1_idx = self.hpo_tree.getHPO2idx_l1(l1_hpo)
                    y_sub = total_model[l1_idx](PhraseDataSet4predictFunc(i, self.fasttext_model)).squeeze()
                    if y_sub.size(0) > 10:
                        prediction_sub = y_sub.topk(10)[1].tolist()
                        scores_p_sub = torch.softmax(y_sub, dim=0).topk(10)[0].tolist()
                    else:
                        prediction_sub = y_sub.topk(y_sub.size(0))[1].tolist()
                        scores_p_sub = torch.softmax(y_sub, dim=0).topk(y_sub.size(0))[0].tolist()
                    Candidate_hpos_sub.update(
                        [total_idx2hpo[l1_idx][prediction_sub[idx]] for idx in range(len(prediction_sub)) if
                         scores_p_sub[idx] >= param2])
                if len(Candidate_hpos_sub) != 0 and "None" not in Candidate_hpos_sub:
                    Candidate_hpos_sub = list(Candidate_hpos_sub)
                    candidate_phrase = [self.hpo_tree.getNameByHPO(item) for item in Candidate_hpos_sub]
                    raw_phrase = i
                    ans_hpo, score, class_num = produceCandidateTripleSlow(raw_phrase, candidate_phrase, self.bert_model,
                                                                       Candidate_hpos_sub, param3)
                    if ans_hpo != "None":
                        res.append([i, ans_hpo])
                        flag = False
            if flag:
                res.append([i, "None"])
        return res
