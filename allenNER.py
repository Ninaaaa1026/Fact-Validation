from allennlp.predictors.predictor import Predictor
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
        files = os.listdir('./wiki/')
        norm_docs = {}
        for f in files:
            with open('./wiki/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    senNum=oneLine[1]
                    norm_sen = ""
                    for word in oneLine[2:]:
                        if word.isalpha() or word.isnumeric():
                            norm_sen+=self.lemmatize(word.lower())+" "
                    if title not in norm_docs.keys():
                        norm_docs[title]=[{'senNum':senNum,'sentence':norm_sen}]
                    else:
                        norm_docs[title].append({'senNum':senNum,'sentence':norm_sen})
        return norm_docs

# given a query and an index returns a list of the k highest scoring documents as tuples containing <docid,score>
def query_tfidf(query, index, k=5):
    # scores stores doc ids and their scores
    scores = Counter()

    N = index.num_docs()
    for term in query:
        docids = index.docids(term)
        for docid in docids:
            if docid not in scores.keys():
                scores[docid] = 0
            i = docids.index(docid)
            scores[docid] += log(1 + index.freqs(term)[i]) * log(N / index.f_t(term))
    for docid, score in scores.items():
        scores[docid] = score / sqrt(index.doc_len[docid])
    return scores.most_common(k)

class SentInvertedIndex:
    def __init__(self, doc_titles, doc_term_freqs,vocab):
        self.vocab = vocab
        self.sent_len = {}
        self.sent_term_freqs = [[] for i in range(len(vocab))]
        self.sent_ids = [[] for i in range(len(vocab))]
        self.sent_freqs = [0] * len(vocab)
        self.total_num_sents = 0
        self.max_sent_len = 0
        for title,score in doc_titles:
            content = doc_term_freqs[title]
            for sentence in content:
                raw_sentence = sentence["sentence"].split(" ")
                sentid = sentence['senNum']
                sentFrequency = Counter(raw_sentence)
                sent_len = sum(sentFrequency.values())
                self.sent_len[(title, sentid)] = sent_len
                self.total_num_sents += 1
                for term, freq in sentFrequency.items():
                    if term in vocab:
                        term_id = vocab[term]
                        self.sent_ids[term_id].append((title, sentid))
                        self.sent_term_freqs[term_id].append(freq)
                        self.sent_freqs[term_id] += 1

    def num_terms(self):
        return len(self.sent_ids)

    def num_sents(self):
        return self.total_num_sents

    def sentids(self, term):
        term_id = self.vocab[term]
        return self.sent_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.sent_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.sent_freqs[term_id]

    def space_in_bytes(self):
        # this function assumes the integers are now bytes
        space_usage = 0
        for doc_list in self.sent_ids:
            space_usage += len(sent_list)
        for freq_list in self.sent_term_freqs:
            space_usage += len(freq_list)
        return space_usage

def sentence_tfidf(query, index, k=5):
    # scores stores doc ids and their scores
    scores = Counter()

    N = index.num_sents()
    for term in query:
        sentids = index.sentids(term)
        for sentid in sentids:
            if sentid not in scores.keys():
                scores[sentid] = 0
            i = sentids.index(sentid)
            scores[sentid] += log(1 + index.freqs(term)[i]) * log(N / index.f_t(term))
    for docid, score in scores.items():
        scores[sentid] = score / sqrt(index.sent_len[sentid])
    return scores.most_common(k)

def docments_retrieval(entities, norm_docs,k=5):
    doc_score=Counter()
    for doctitle, sentences in norm_docs.items():
        for sentence in sentences:
            for entity in entities:
                if sentence['sentence'].count(entity)>0:
                    print(sentence['sentence'].count(entity))
                    print(sentence['sentence'])
                doc_score.update({doctitle:sentence['sentence'].count(entity)})
    return sorted(doc_score.items(), key=lambda kv: kv[1])[:20]

def sentRetrive(lemmatized_query,topDocTitile):
    unique_token = 0
    query_vocab = {}
    for word in lemmatized_query:
        if word not in query_vocab:
            query_vocab[word]=unique_token
            unique_token += 1
    print(query_vocab)
    #sentences retrieval
    sent_index = SentInvertedIndex(topDocTitile, norm_docs, query_vocab)
    return sentence_tfidf(lemmatized_query, sent_index)

def entityRetrieval(query):
    predicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    results = predicts.predict_json({"sentence": query})
    entity = []
    wordList=[]
    flag = False
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            wordList.append(process().lemmatize(results['words'][index].lower()))
            if str(tag).startswith('B-'):
                phrase = process().lemmatize(results['words'][index].lower())
                flag = True
            elif flag:
                phrase += " " + process().lemmatize(results['words'][index].lower())
                if str(tag).startswith('L-'):
                    flag = False
                    entity.append(phrase)
            else:
                entity.append(process().lemmatize(results['words'][index].lower()))
    return entity,wordList

if __name__ == '__main__':
    query = "Henderson contributed a track with Big Boss"

    # entity retrieval
    entity,wordList=entityRetrieval(query)
    print(entity)
    time1=time.time()

    # data propressing
    proprocess=process()
    norm_docs=proprocess.dataProcessing()

    # docments retrieval
    topDocTitile = docments_retrieval(entity, norm_docs)
    print(topDocTitile)

    # sentences retrieval
    # queryToken = nltk.word_tokenize(query)
    # lemmatized_query = [proprocess.lemmatize(word.lower()) for word in queryToken]
    evidence = sentRetrive(wordList,topDocTitile)
    time2 = time.time()-time1
    print(evidence)
    print(time2)
