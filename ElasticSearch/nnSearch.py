import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk,time

url = "http://localhost:9200/_search"

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, 'a')
    return lemma

def docSelection(url, entityQuery):
    """Simple Elasticsearch Query"""
    query = json.dumps({
        "query": {
            "match": {
                 "title": entityQuery,
            }
        },
        "size": 100
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    # print(results)
    return results

def senSelection(url,query,entityQuery):
    """Simple Elasticsearch Query"""
    # print(titles)
    query = json.dumps({
        "query": {
            "bool": {
                "must": [
                    { "match": {
                        "sentence_text": query
                    }}
                    ,
                    {"match": {
                            "title": entityQuery
                        }
                }
                ]
                # ,
                # "filter": {
                #     "bool": {
                #         "must": titles
                #     }
                # }
            }
        },
        "size": 5
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    print(results)
    return results

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

#not lemmitized
def entityRetrieval2(query):
    results = predicts.predict_json({"sentence": query})
    entity = []
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            entity.append(results['words'][index])
    return entity

def entityRetrieval3(query):
    results = predicts.predict_json({"sentence": query})
    entity = []
    wordList=[]
    flag = False
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            wordList.append(results['words'][index].lower())
            if str(tag).startswith('B-'):
                phrase = results['words'][index].lower()
                flag = True
            elif flag:
                phrase += " " + results['words'][index].lower()
                if str(tag).startswith('L-'):
                    flag = False
                    entity.append(phrase)
            else:
                entity.append(results['words'][index].lower())
    return entity,wordList

if __name__ == '__main__':
    predicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    with open('./devset100.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    nn = nnClassifier()
    nn.restoreNet()

    fullResult = {}
    time1=time.time()
    for key, content in d.items():
        claim = content['claim']
        # print(claim)
        normClaim=" ".join([lemmatize(word.lower()) for word in claim.split(" ")])
        query=" ".join(entityRetrieval2(claim))
        print(query)
        # print(normClaim)
        # print(claim)
        evidenceList=[]
        sentence=""
        # docTitle=set()
        # for docs in docSelection(url, query)['hits']['hits']:
        #     docTitle.update([docs['_source']["page_identifier"]])
        # titleList=[]
        # for title in docTitle:
        #     titleList.append({"term":{"page_identifier":title}})
        for candidate in senSelection(url,claim,query)['hits']['hits']:
            # if evidenceList == []:
            #     sentence = candidate['_source']["sentence_text"]
            pro.extend(predict(normClaim, candidate['_source']["sentence_text"]))
            evidenceList.append([candidate['_source']["page_identifier"],int(candidate['_source']["sentence_number"])])
        nn.classify(pro)
        # print(evidenceList)
        # if evidenceList!=[]:
        #     sentence=evidenceList[0][2]
        judge = predict(normClaim, sentence)
        fresult = {}
        fresult['claim'] = content['claim']
        fresult['label'] = judge
        fresult['evidence'] = evidenceList
        fullResult[key] = fresult

    # store result
    with open('./tfidf.json', 'w', encoding='utf-8') as f:
        json.dump(fullResult, f)
        f.close()
