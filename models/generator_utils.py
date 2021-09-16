import logging
from functools import partial

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config
from question_generator.questiongenerator import QuestionGenerator
import spacy
import nltk.data

logger = logging.getLogger(__name__)


# check that there is at least one verb, noun, adjectives/adverb
def answer_is_full(answer, pipeline_model):
    parsed = pipeline_model(answer)
    noun_phrases = [chunk.text for chunk in parsed.noun_chunks]
    d = {}
    for token in parsed:
        d.setdefault(token.pos_, []).append(token.lemma_)
    logger.debug('')
    return (len(noun_phrases) > 0 or len(d.get('VERB', [])) > 0) and (
            len(d.get('ADJ', [])) > 0 or len(d.get('ADV', [])) > 0)


def answer_is_full_init():
    model = spacy.load("en_core_web_sm")
    return partial(answer_is_full, pipeline_model=model)


def generate_questions(context, amount, question_model):
    return list(map(lambda el: el['question'],
                    question_model.generate(article=context,
                                            num_questions=amount,
                                            answer_style="sentences")))


def generate_questions_init():
    model = QuestionGenerator()
    return partial(generate_questions, question_model=model)


def get_summary(text, tokenizer, model, device, sent_detector):
    preprocess_text = text.strip().replace("\n", "")
    t5_prepared_Text = "summarize: " + preprocess_text
    tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)
    summary_ids = model.generate(tokenized_text,
                                 num_beams=4,
                                 no_repeat_ngram_size=2,
                                 min_length=3,
                                 max_length=30,
                                 early_stopping=True)
    output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    # capitalize the first letter
    output = output[0].upper() + output[1:]
    # break the string into array of sentences
    sentences = sent_detector.tokenize(output.strip())
    # return the first sentence
    return sentences[0]


def get_summary_init(device='cpu'):
    nltk.download('punkt')
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    model = T5ForConditionalGeneration.from_pretrained('t5-small')
    tokenizer = T5Tokenizer.from_pretrained('t5-small')
    d = torch.device("cuda" if device != 'cpu' and torch.cuda.is_available() else "cpu")
    return partial(get_summary, tokenizer=tokenizer, model=model, device=d, sent_detector=sent_detector)


def main():
    test_string = """Somebody once told me the world is gonna roll me
    I aint the sharpest tool in the shed
    She was looking kind of dumb with her finger and her thumb
    In the shape of an "L" on her forehead"""
    print(answer_is_full_init()(test_string))
    print(generate_questions_init()(test_string, 3))
    print(get_summary_init()(test_string))


if __name__ == "__main__":
    main()
