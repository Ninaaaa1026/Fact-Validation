import nltk
import os
from collections import Counter

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
        norm_docs = {}
        doc_vocab = {}
        unique_token = 0
        for f in files:
            with open('./wiki-pages-text/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    senNum=oneLine[1]
                    # stemming and lower
                    cnt = Counter()
                    # raw_sentence=[ ]
                    for word in oneLine[2:]:
                        if word.isalpha():
                            w=self.lemmatize(word.lower())
                            cnt[w] += 1
                            doc_vocab[w]=unique_token
                            unique_token+=1
                    # if title not in norm_docs.keys():
                    #     norm_docs[title]=[{'senNum':senNum,'sentence':raw_sentence,'frequency':cnt}]
                    # else:
                    #     norm_docs[title].append({'senNum':senNum,'sentence':raw_sentence,'frequency':cnt})
                    if title not in norm_docs.keys():
                        norm_docs[title]=[{'senNum':senNum,'frequency':cnt}]
                    else:
                        norm_docs[title].append({'senNum':senNum,'frequency':cnt})
        return norm_docs,doc_vocab
