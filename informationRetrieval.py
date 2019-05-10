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
        files = os.listdir('./wiki-pages-text/')
        norm_docs = {}
        doc_vocab = {}
        unique_token = 0
        count=0
        doc_id_pair={}
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
def query_tfidf(query, index, k=10):
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

def vbyte_encode(num):
    # out_bytes stores a list of output bytes encoding the number
    out_bytes = []
    while num >= 128:
        out_bytes.append(num % 128)
        num = int(num / 128)
    out_bytes.append(num + 128)
    return out_bytes

def vbyte_decode(input_bytes, idx):
    # x stores the decoded number
    x = 0
    # consumed stores the number of bytes consumed to decode the number
    consumed = 0
    y = input_bytes[idx]
    while y < 128:
        x += y * 128 ** consumed
        consumed += 1
        y = input_bytes[idx + consumed]
    x += (y - 128) * 128 ** consumed
    consumed += 1
    return x, consumed

def decompress_list(input_bytes, gapped_encoded):
    res = []
    prev = 0
    idx = 0
    while idx < len(input_bytes):
        dec_num, consumed_bytes = vbyte_decode(input_bytes, idx)
        idx += consumed_bytes
        num = dec_num + prev
        res.append(num)
        if gapped_encoded:
            prev = num
    return res

class DocCompressedInvertedIndex:
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

        for term_id in vocab.values():
            doc_ids = vbyte_encode(self.doc_ids[term_id][0])
            doc_term_freqs = vbyte_encode(self.doc_term_freqs[term_id][0])
            for i in range(1, self.doc_freqs[term_id]):
                num_doc_ids = self.doc_ids[term_id][i] - self.doc_ids[term_id][i - 1]
                doc_ids.extend(vbyte_encode(num_doc_ids))
                doc_term_freqs.extend(vbyte_encode(self.doc_term_freqs[term_id][i]))
            self.doc_ids[term_id] = doc_ids
            self.doc_term_freqs[term_id] = doc_term_freqs

    def num_terms(self):
        return len(self.doc_ids)

    def num_docs(self):
        return self.total_num_docs

    def docids(self, term):
        term_id = self.vocab[term]
        # We decompress
        return decompress_list(self.doc_ids[term_id], True)

    def freqs(self, term):
        term_id = self.vocab[term]
        # We decompress
        return decompress_list(self.doc_term_freqs[term_id], False)

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

# def get_key(dict, value):
#     titles=[]
#     for rank in value:
#         for k, v in dict.items():
#             if v['DOCID'] == rank[0]:
#                 titles.append(k)
#     return titles

if __name__ == '__main__':
    # query = sys.argv[0]
    query = "Henderson contributed a track with Big Boss"

    # data propressing
    proprocess=process()
    norm_docs,doc_vocab,doc_id_pair=proprocess.dataProcessing()

    #docments retrieval
    compressed_index = DocCompressedInvertedIndex(norm_docs,doc_vocab)
    queryToken=nltk.word_tokenize(query)
    lemmatized_query = [proprocess.lemmatize(word.lower()) for word in queryToken]
    topDocID = query_tfidf(lemmatized_query, compressed_index)
    topDocTitile=[doc_id_pair[item[0]] for item in topDocID]
    print(topDocTitile)