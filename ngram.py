import random
import re
from string import punctuation


def preprocess(text):
    sentences_ = []
    for sentence in re.split('[.!?] (?=[A-ZА-Я])|\n', text):
        sentences_.append(
            [word.lower() for word in sentence.translate(str.maketrans('', '', punctuation)).split(' ') if
             word.isalpha()])
    return sentences_


class NGramModel:

    def __init__(self, corpus=None):
        self.vocab_ = set()
        self.tri_gram_counts = {}
        self.bi_gram_counts = {}
        if corpus:
            self.__generate_vocab(corpus)
            self.__bi_gram(corpus)
            self.__tri_gram(corpus)

    def fit(self, corpus):
        self.__generate_vocab(corpus)
        self.__bi_gram(corpus)
        self.__tri_gram(corpus)

    def generate(self, context, length):
        text = ''
        for i in range(length):
            text += ' ' + self.predict_next_word(context + text)
        return text

    def __bi_gram(self, corpus):
        for sentence in corpus:
            for i in range(len(sentence) - 1):
                bi_gram_ = (sentence[i], sentence[i + 1])

                if bi_gram_ in self.bi_gram_counts.keys():
                    self.bi_gram_counts[bi_gram_] += 1
                else:
                    self.bi_gram_counts[bi_gram_] = 1

    def __tri_gram(self, corpus):
        for sentence in corpus:
            for i in range(len(sentence) - 2):
                tri_gram_ = (sentence[i], sentence[i + 1], sentence[i + 2])

                if tri_gram_ in self.tri_gram_counts.keys():
                    self.tri_gram_counts[tri_gram_] += 1
                else:
                    self.tri_gram_counts[tri_gram_] = 1

    def __generate_vocab(self, corpus):
        for sentence in corpus:
            self.vocab_.update(set(sentence))
        return self.vocab_

    def predict_next_word(self, input_):
        top = self.predict_prob_next_word(input_)
        if not top:
            return random.choice(list(self.vocab_))
        return random.choice(top)[0]

    def predict_prob_next_word(self, input_):
        tokenized_input = input_.lower().split(' ')
        if len(tokenized_input) < 2:
            tokenized_input.append(random.choice(list(self.vocab_)))
        last_bi_gram = tokenized_input[-2:]

        vocab_probabilities = {}

        test_bi_gram = (last_bi_gram[0], last_bi_gram[1])
        test_bi_gram_count = self.bi_gram_counts.get(test_bi_gram, 1)

        for vocab_word in self.vocab_:
            test_tri_gram = (last_bi_gram[0], last_bi_gram[1], vocab_word)
            test_tri_gram_count = self.tri_gram_counts.get(test_tri_gram, 0)

            probability = test_tri_gram_count / test_bi_gram_count
            vocab_probabilities[vocab_word] = probability

        top_suggestions = [prob for prob
                           in sorted(vocab_probabilities.items(), key=lambda x: x[1], reverse=True)
                           if prob[1] > 0]
        return top_suggestions
