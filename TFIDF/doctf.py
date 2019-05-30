import json
# from allennlp.service.predictors.predictor import Predictor
import nltk
import os,sys
from collections import Counter,defaultdict
from math import log, sqrt
import time


class process():
    def __init__(self):
        self.lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

    def lemmatize(self,word):
        lemma = self.lemmatizer.lemmatize(word, 'v')
        if lemma == word:
            lemma = self.lemmatizer.lemmatize(word, 'n')
            if lemma == word:
                lemma = self.lemmatizer.lemmatize(word, 'a')
        return lemma

    def dataProcessing(self):
        # Save to the current directory
        files = os.listdir('./wiki-pages-text/')
        doc_vocab = defaultdict(Counter)
        for fname in files:
            with open('./wiki-pages-text/'+ fname, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine = raw_doc.split(" ")
                    title = oneLine[0]

                    for word in oneLine[2:]:
                        doc_vocab[title][self.lemmatize(word.lower())] += 1
                f.close()
        return doc_vocab

if __name__ == '__main__':

    proprocess = process()
    # data propressing

    doc_vocab = proprocess.dataProcessing()

    with open('./doc_vocab.json', 'w', encoding='utf-8') as f:
        json.dump(doc_vocab, f)
        f.close()
