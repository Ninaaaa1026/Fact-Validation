import nltk
import os,sys
from collections import Counter
from math import log, sqrt

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

    def counterAdd(self,x,y):
        for k, v in y.items():
            if k in x.keys():
                x[k] += v
            else:
                x[k] = v
        return x

    def dataProcessing(self):
        # Save to the current directory
        files = os.listdir('./wiki/')
        norm_docs = {}
        doc_vocab = {}
        unique_token = 0
        count=0
        doc_id_pair={}
        for f in files:
            with open('./wiki/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    senNum=oneLine[1]
                    # stemming and lower
                    cnt = Counter()
                    # raw_sentence=[ ]
                    for word in oneLine[2:]:
                        if word.isalpha() or word.isnumeric():
                            w=self.lemmatize(word.lower())
                            cnt[w] += 1
                            if w not in doc_vocab:
                                doc_vocab[w]=unique_token
                                unique_token+=1
                    if title not in norm_docs.keys():
                        norm_docs[title]={'DOCID':count,'docFrequency':cnt,'senFrequency':[{'senNum':senNum,'frequency':cnt}]}
                        doc_id_pair[count]=title
                        count+=1
                    else:
                        norm_docs[title]['senFrequency'].append({'senNum':senNum,'frequency':cnt})
                        norm_docs[title]['docFrequency']=self.counterAdd(norm_docs[title]['docFrequency'],cnt)
        return norm_docs,doc_vocab,doc_id_pair

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
            print(scores[sentid])
            scores[sentid] += log(1 + index.freqs(term)[i]) * log(N / index.f_t(term))
    for docid, score in scores.items():
        scores[sentid] = score / sqrt(index.sent_len[sentid])
    return scores.most_common(k)

class DocInvertedIndex:
    def __init__(self, doc_term_freqs,vocab):
        self.vocab = vocab
        self.doc_len = [0] * len(doc_term_freqs)
        self.doc_term_freqs = [[] for i in range(len(vocab))]
        self.doc_ids = [[] for i in range(len(vocab))]
        self.doc_freqs = [0] * len(vocab)
        self.total_num_docs = 0
        self.max_doc_len = 0
        for docTitle, content in doc_term_freqs.items():
            docid=content["DOCID"]
            docFrequency=content["docFrequency"]
            doc_len = sum(docFrequency.values())
            self.max_doc_len = max(doc_len, self.max_doc_len)
            self.doc_len[docid] = doc_len
            self.total_num_docs += 1
            for term, freq in docFrequency.items():
                term_id = vocab[term]
                if len(self.doc_ids)<term_id:
                    print(len(self.doc_ids),term_id)
                self.doc_ids[term_id].append(docid)
                self.doc_term_freqs[term_id].append(freq)
                self.doc_freqs[term_id] += 1

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        return self.doc_ids[term_id]

    def freqs(self, term):
        term_id = self.vocab[term]
        return self.doc_term_freqs[term_id]

    def f_t(self, term):
        term_id = self.vocab[term]
        return self.doc_freqs[term_id]

    def space_in_bytes(self):
        # this function assumes the integers are now bytes
        space_usage = 0
        for doc_list in self.doc_ids:
            space_usage += len(doc_list)
        for freq_list in self.doc_term_freqs:
            space_usage += len(freq_list)
        return space_usage

class SentInvertedIndex:
    def __init__(self, doc_term_freqs,vocab):
        self.vocab = vocab
        self.sent_len = {}
        self.sent_term_freqs = [[] for i in range(len(vocab))]
        self.sent_ids = [[] for i in range(len(vocab))]
        self.sent_freqs = [0] * len(vocab)
        self.total_num_sents = 0
        self.max_sent_len = 0
        for docTitle, content in doc_term_freqs.items():
            docid = content["DOCID"]
            sentences = content["senFrequency"]
            for sentence in sentences:
                sentid = sentence['senNum']
                sentFrequency = sentence['frequency']
                sent_len = sum(sentFrequency.values())
                self.sent_len[(docid, sentid)] = sent_len
                self.total_num_sents += 1
                for term, freq in sentFrequency.items():
                    term_id = vocab[term]
                    self.sent_ids[term_id].append((docid, sentid))
                    self.sent_term_freqs[term_id].append(sentFrequency)
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


if __name__ == '__main__':
    # query = sys.argv[0]
    query = "Henderson contributed a track with Big Boss"

    # data propressing
    proprocess=process()
    norm_docs,doc_vocab,doc_id_pair=proprocess.dataProcessing()

    '''
    #docments retrieval
    compressed_index = DocInvertedIndex(norm_docs,doc_vocab)
    queryToken=nltk.word_tokenize(query)
    lemmatized_query = [proprocess.lemmatize(word.lower()) for word in queryToken]
    topDocID = query_tfidf(lemmatized_query, compressed_index)
    topDocTitile=[doc_id_pair[item[0]] for item in topDocID]
    print(topDocTitile)
    '''

    #sentences retrieval
    index = SentInvertedIndex(norm_docs,doc_vocab)
    queryToken=nltk.word_tokenize(query)
    lemmatized_query = [proprocess.lemmatize(word.lower()) for word in queryToken]
    topSent = sentence_tfidf(lemmatized_query, index)
    print(topSent)
