# -*- coding: utf8  -*-
import os.path
import pickle

import ngram

if __name__ == '__main__':

    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        model = ngram.NGramModel()
        text = []
        with open('text.txt', 'r', encoding='utf-8') as file:
            for line in file:
                text.append(ngram.preprocess(line)[0])

        model.fit(text)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)

    pred = model.generate('потому что', 3)
    print(pred)
