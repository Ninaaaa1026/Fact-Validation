from whoosh.index import create_in
from whoosh.fields import *
import nltk
import os,sys,time
from whoosh.qparser import QueryParser
from whoosh.index import open_dir


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
        writer = ix.writer()

        for f in files:
            with open('./wiki/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    titleString=""
                    for item in title.split("_"):
                        titleString+=item+" "
                    senNum=oneLine[1]
                    norm_sen = ""
                    writer.add_document(title=u"" + title, path=u""+title, content=u"" + titleString)

                    for word in oneLine[2:]:
                        if word.isalpha() or word.isnumeric():
                            lem_word = self.lemmatize(word.lower())
                            norm_sen+= lem_word +" "
                    writer.add_document(title=u""+title, path=u""+senNum,content=u""+norm_sen)
        writer.commit()
        print()


schema = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))
if not os.path.exists("indexer"):
    os.mkdir("indexer")
    ix= create_in("indexer",schema)
    time1=time.time()
    process().dataProcessing()
    print(time.time()-time1)
else:
    ix = open_dir("indexer")

with ix.searcher() as searcher:
    qstr="Henderson contributed Big Boss"
    q=""
    for word in qstr.split(" "):
        q+=process().lemmatize(word.lower() )+" "

    query = QueryParser("content", ix.schema).parse(q)
    results = searcher.search(query)
    time1=time.time()
    for result in results:
        print(result)
    print(time.time() - time1)