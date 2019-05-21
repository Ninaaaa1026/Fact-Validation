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

    def dataProcessing(self,num):
        # Save to the current directory
        files = os.listdir('./wiki2/')
        # if num == 1:
        writer = ix1.writer()
        # if num == 2:
        #     writer = ix2.writer()
        # if num == 3:
        #     writer = ix3.writer()
        # if num == 0:
        #     writer = ix4.writer()
        for idx,f in enumerate(files):
            # if (idx+1) % 4 == num:
            print(idx)
            time1 = time.time()
            with open('./wiki2/'+ f, 'r', encoding='utf-8') as f:
                for raw_doc in f:
                    oneLine=raw_doc.split(" ")
                    title=oneLine[0]
                    titleString=title.replace("_"," ")
                    senNum=oneLine[1]
                    norm_sen = ""
                    for word in oneLine[2:]:
                        # if word.isalpha() or word.isnumeric():
                        lem_word = self.lemmatize(word.lower())
                        norm_sen+= lem_word +" "
                        # writer1.add_document(title=u"" + title, path=u"" + title, content=u"" + titleString)
                    writer.add_document(title=u"" + title, senNum=u"" + senNum, content=u"" + norm_sen)
            writer.commit()
            writer = ix1.writer()
            print(time.time() - time1)


schema1 = Schema(title=TEXT(phrase=False), senNum=ID, content=TEXT(stored=True))
# schema2 = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))
# schema3 = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))
# schema4 = Schema(title=TEXT(stored=True), path=ID(stored=True), content=TEXT(stored=True))

if not os.path.exists("indexer2"):
    os.mkdir("indexer2")
    ix1= create_in("indexer2", schema1)
    process().dataProcessing(1)
else:
    ix1 = open_dir("indexer2")
#
# if not os.path.exists("indexer2"):
#     os.mkdir("indexer2")
#     ix2= create_in("indexer2", schema2)
#     time1=time.time()
#     process().dataProcessing(2)
#     print(time.time()-time1)
# else:
#     ix2 = open_dir("indexer2")
#
# if not os.path.exists("indexer3"):
#     os.mkdir("indexer3")
#     ix3 = create_in("indexer3", schema3)
#     time1 = time.time()
#     process().dataProcessing(3)
#     print(time.time() - time1)
# else:
#     ix3 = open_dir("indexer3")
#
# if not os.path.exists("indexer4"):
#     os.mkdir("indexer4")
#     ix4= create_in("indexer4", schema4)
#     time1=time.time()
#     process().dataProcessing(0)
#     print(time.time()-time1)
# else:
#     ix4 = open_dir("indexer4")

qstr="Salgaocar F.C. won the cup"
q=""
for word in qstr.split(" "):
    q+=process().lemmatize(word.lower() )+" "

result=[]
with ix1.searcher() as searcher1:
    query = QueryParser("content", ix1.schema).parse(q)
    results = searcher1.search(query)
    # results = searcher1.document()
    time1=time.time()
    for r in results:
        result.append({"title": r["title"], "senNum": r["senNum"], "content": r["content"]})

# with ix2.searcher() as searcher2:
#     query = QueryParser("content", ix2.schema).parse(q)
#     # results = searcher2.search(query)
#
#     time1=time.time()
#     for r in results:
#         result.append({"title":r["title"],"senNum":r["path"],"content":r["content"]})
#
# with ix3.searcher() as searcher3:
#     query = QueryParser("content", ix3.schema).parse(q)
#     results = searcher3.search(query)
#     time1=time.time()
#     for r in results:
#         result.append({"title": r["title"], "senNum": r["path"], "content": r["content"]})
#
# with ix4.searcher() as searcher4:
#     # query = QueryParser("content", ix4.schema).parse(q)
#     # results = searcher4.search(query)
#     results = searcher4.document()
#     time1=time.time()
#     for r in results:
#         result.append({"title": r["title"], "senNum": r["path"], "content": r["content"]})

for r in result:
    print(r)
