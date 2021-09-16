import logging

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from classifier_abstract import ClassifierAbstract

logger = logging.getLogger(__name__)


class SingleEmotionClassifier(ClassifierAbstract):
    def __init__(self, device='cpu'):
        self._description = "mrm8488/t5-base-finetuned-emotion"

        self._device = torch.device("cuda" if device != 'cpu' and torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained("mrm8488/t5-base-finetuned-emotion", use_fast=False)
        self._model = AutoModelForSeq2SeqLM.from_pretrained("mrm8488/t5-base-finetuned-emotion")
        self._model.to(self._device)
        logger.info('%s initialized using %s', self.__class__.__name__, self._device)

    @property
    def description(self):
        return self._description

    def classify(self, text):
        input_ids = self._tokenizer.encode(text + '</s>', return_tensors='pt')
        output = self._model.generate(input_ids=input_ids, max_length=5)
        dec = [self._tokenizer.decode(ids) for ids in output]
        label = dec[0]

        # remove pad token prefix and eos token suffix
        prefix = '<pad>'
        suffix = '</s>'
        if label.startswith(prefix):
            label = label[len(prefix):]
        if label.endswith(suffix):
            label = label[:-len(suffix)]

        logger.debug('%s classify: "%s ... %s" => %s',
                     self.__class__.__name__, text[:min(15, max(len(text) - 15, 0))], text[-15:], label)

        return label


def main():
    classifier = SingleEmotionClassifier()
    test_string = """Somebody once told me the world is gonna roll me
    I aint the sharpest tool in the shed
    She was looking kind of dumb with her finger and her thumb
    In the shape of an "L" on her forehead"""
    print(classifier.classify(test_string))


if __name__ == "__main__":
    main()
