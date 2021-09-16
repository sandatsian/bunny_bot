import json
import logging
import os

import tez
import transformers
import torch
import torch.nn as nn
from sklearn import metrics
from transformers import AdamW, get_linear_schedule_with_warmup

from classifier_abstract import ClassifierAbstract

logger = logging.getLogger(__name__)


class EmotionClassifierModel(tez.Model):
    def __init__(self, num_train_steps, num_classes):
        super().__init__()
        self.bert = transformers.SqueezeBertModel.from_pretrained("squeezebert/squeezebert-uncased")
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, num_classes)
        self.num_train_steps = num_train_steps
        self.step_scheduler_after = "batch"

    def fetch_optimizer(self):
        param_optimizer = list(self.named_parameters())
        no_decay = ["bias", "LayerNorm.bias"]
        optimizer_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.001,
            },
            {
                "params": [
                    p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        opt = AdamW(optimizer_parameters, lr=3e-5)
        return opt

    def fetch_scheduler(self):
        sch = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=0, num_training_steps=self.num_train_steps
        )
        return sch

    def loss(self, outputs, targets):
        if targets is None:
            return None
        return nn.BCEWithLogitsLoss()(outputs, targets.float())

    def monitor_metrics(self, outputs, targets):
        if targets is None:
            return {}

        outputs = torch.sigmoid(outputs)
        outputs = outputs.cpu().detach().numpy()
        targets = targets.cpu().detach().numpy()

        fpr_micro, tpr_micro, _ = metrics.roc_curve(targets.ravel(), outputs.ravel())
        auc_micro = metrics.auc(fpr_micro, tpr_micro)
        return {"auc": auc_micro}

    def forward(self, ids, mask, targets=None):
        o_2 = self.bert(ids, attention_mask=mask)["pooler_output"]
        b_o = self.bert_drop(o_2)
        output = self.out(b_o)
        loss = self.loss(output, targets)
        acc = self.monitor_metrics(output, targets)
        return output, loss, acc


class MultipleEmotionsClassifier(ClassifierAbstract):
    def __init__(self, topn=5, device='cpu'):
        self._description = "custom trained model on 'goemotions' dataset using 'squeezebert'"

        self._topn = topn
        self._device = torch.device("cuda" if device != 'cpu' and torch.cuda.is_available() else "cpu")

        # opens file with labels that supposed to be in the same dir as this script
        dir_path = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(dir_path, 'emotions_mapping.json'), 'r') as json_file:
            self._mapping = json.load(json_file)

        self._tokenizer = transformers.SqueezeBertTokenizer.from_pretrained(
            "squeezebert/squeezebert-uncased", do_lower_case=True
        )
        self._model = EmotionClassifierModel(0, len(self._mapping))
        self._model.load(os.path.join(dir_path, 'model.bin'), device=self._device)
        logger.info('%s: initialized using %s', self.__class__.__name__, self._device)

    @property
    def description(self):
        return self._description

    def classify(self, text):
        max_len = 35
        with torch.no_grad():
            inputs = self._tokenizer.encode_plus(text,
                                                 None,
                                                 add_special_tokens=True,
                                                 max_length=max_len,
                                                 padding="max_length",
                                                 truncation=True)
            ids = inputs["input_ids"]
            ids = torch.LongTensor(ids, device=self._device).unsqueeze(0)

            attention_mask = inputs["attention_mask"]
            attention_mask = torch.LongTensor(attention_mask, device=self._device).unsqueeze(0)

            output = self._model.forward(ids, attention_mask)[0]
            output = torch.sigmoid(output)

            probas, indices = torch.sort(output)

        probas = probas.cpu().numpy()[0][::-1]
        indices = indices.cpu().numpy()[0][::-1]

        result = {}
        for i, p in zip(indices[:self._topn], probas[:self._topn]):
            result.setdefault(self._mapping[str(i)], p)

        logger.debug('%s classify: "%s ... %s" => %s',
                     self.__class__.__name__, text[:min(15, max(len(text) - 15, 0))], text[-15:], result)

        return result


def main():
    classifier = MultipleEmotionsClassifier()
    test_string = """Somebody once told me the world is gonna roll me
    I aint the sharpest tool in the shed
    She was looking kind of dumb with her finger and her thumb
    In the shape of an "L" on her forehead"""
    print(classifier.classify(test_string))


if __name__ == "__main__":
    main()
