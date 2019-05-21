import json
from allennlp.service.predictors.predictor import Predictor
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
        vocab = Counter()
        doc_vocab = defaultdict(Counter)
        norm_docs = defaultdict(lambda:defaultdict())
        num_sentence = 0
        for f in files:
            with open('./wiki-pages-text/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    senNum=oneLine[1]
                    norm_sen = ""
                    num_sentence += 1

                    for word in oneLine[2:]:
                        if word.isalpha() or word.isnumeric():
                            lem_word = self.lemmatize(word.lower())
                            norm_sen+= lem_word +" "
                            vocab[lem_word] += 1
                            doc_vocab[title][lem_word] += 1

                    # if title not in norm_docs.keys():
                    norm_docs[title][senNum]=norm_sen
        return norm_docs, vocab, doc_vocab, num_sentence

# given a query and an index returns a list of the k highest scoring documents as tuples containing <docid,score>
def query_tfidf(vocab, query, index, k=5):
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
            for senNum,sentence in content.items():
                raw_sentence = sentence.split(" ")
                sentid = senNum
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

def sentence_tfidf(num_sentence, vocab, query, index, k=5):
    # scores stores doc ids and their scores
    scores = Counter()

    for term in query:
        sentids = index.sentids(term)
        for sentid in sentids:
            if sentid not in scores.keys():
                scores[sentid] = 0
            i = sentids.index(sentid)
            scores[sentid] += log(1 + index.freqs(term)[i]) * log(num_sentence / vocab[term])

    for sentid, score in scores.items():
        scores[sentid] = score / sqrt(index.sent_len[sentid])


    return scores.most_common(k)

def docments_retrieval(entities, norm_docs,k=5):
    doc_score=Counter()
    for doctitle in doc_vocab.keys():

        for entity in entities:
            if entity in doctitle.replace("_"," "):
                doc_score[doctitle] += 1
            # if entity in doc_vocab[doctitle].keys():
            #     doc_score.update({doctitle:doc_vocab[doctitle][entity]})
    return doc_score.most_common(k)

def sentRetrive(lemmatized_query,topDocTitile, vocab, num_sentence):
    unique_token = 0
    query_vocab = {}
    for word in lemmatized_query:
        if word not in query_vocab:
            query_vocab[word]=unique_token
            unique_token += 1
    #print(query_vocab)
    #sentences retrieval
    sent_index = SentInvertedIndex(topDocTitile, norm_docs, query_vocab)
    return sentence_tfidf(num_sentence, vocab, lemmatized_query, sent_index)

def entityRetrieval(query):
    results = predicts.predict_json({"sentence": query})
    entity = []
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            entity.append(process().lemmatize(results['words'][index].lower()))
    return entity

#not lemmitized
def entityRetrieval2(query):
    results = predicts.predict_json({"sentence": query})
    entity = []
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            entity.append(results['words'][index])
    return entity

def predict(query, sentence):
    prediction = predictor.predict(
        hypothesis=query,
        premise=sentence
    )
    p_pos = prediction['label_probs'][0]
    p_neg = prediction['label_probs'][1]
    p_neu = prediction['label_probs'][2]
    max_prob = max(p_pos, p_neg, p_neu)
    if max_prob == p_pos:
        return 'SUPPORTS'
    elif max_prob == p_neg:
        return 'REFUTES'
    else:
        return 'NOT ENOUGH INFO'


if __name__ == '__main__':
    time0=time.time()
    with open('./mytest.json', 'r', encoding='utf-8') as f:
        test = json.load(f)
        f.close()


    proprocess = process()
    # data propressing

    norm_docs, vocab, doc_vocab, num_sentence = proprocess.dataProcessing()
    store = {}
    store['vocab'] = vocab
    store['num_sentence'] = num_sentence

    #store vocab
    with open('./vocab.json', 'w', encoding='utf-8') as f:
        json.dump(store, f)
        f.close()

    # store normdoc
    with open('./normdoc.json', 'w', encoding='utf-8') as f:
        json.dump(norm_docs, f)
        f.close()

    # #load stored normdoc
    # with open('./normdocsmall.json', 'r', encoding='utf-8') as f:
    #     d = json.load(f)
    #     f.close()
    #
    # norm_docs = d['norm_docs']
    # vocab = d['vocab']
    # doc_vocab = d['doc_vocab']
    # num_sentence = d['num_sentence']


    fullResult = {}

    time1 = time.time()
    print('process',time1-time0)

    predicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")

    time2 = time.time()
    print('load', time2  - time1)

    for title, content in test.items():
        time3 = time.time()
        #print('start',time3  - time2)
        query = content['claim']
        # entity retrieval
        entity = entityRetrieval2(query)
        # entity2 = entityRetrieval2(query)
        # print(entity2)

        time3_1 = time.time()
        print('ner', time3_1 - time3)
        # docments retrieval
        topDocTitle = docments_retrieval(entity, norm_docs)
        time4 = time.time()
        print('doc',  time4 - time3_1)

        if topDocTitle == []:
            judge = 'NOT ENOUGH INFO'
        else:
            # sentences retrieval
            queryToken = nltk.word_tokenize(query)
            lemmatized_query = []
            string_query = ''
            for word in queryToken:
                word1 = proprocess.lemmatize(word.lower())
                lemmatized_query.append(word1)
                string_query += word1 + ' '

            evidence = sentRetrive(lemmatized_query, topDocTitle, vocab, num_sentence)
            time5 = time.time()
            print('sent', time5 - time4)

            evidenceList = []

            # major vote
            # result = Counter()
            # for (docTitle, senNum), score in evidence:
            #     evidenceList.append([docTitle, int(senNum)])
            #     sentence = norm_docs[docTitle][senNum]
            #     result[predict(string_query, sentence)] += 1
            # [(judge, count)] = result.most_common(1)

            # top one
            print(evidence)
            for (docTitle, senNum), score in evidence:
                evidenceList.append([docTitle, int(senNum)])
            (docTitle, senNum), score = evidence[0]
            sentence = norm_docs[docTitle][senNum]
            # print(string_query)
            # print(sentence)
            judge = predict(string_query, sentence)

        fresult = {}
        fresult['claim'] = content['claim']
        fresult['label'] = judge
        fresult['evidence'] = evidenceList

        fullResult[title] = fresult
        time2 = time.time()
        print('judge', time2 - time5)

    print(fullResult)
