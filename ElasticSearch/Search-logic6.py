import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk,time
from collections import Counter
import re
url = "http://localhost:9200/collections/_search"

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

def senSelection(url,normclaim,claim,entities):
    """Simple Elasticsearch Query"""
    # print(titles)

    query = json.dumps({
        "query": {
            "bool": {
                "should": entities,
                "must":[{ "match": {
                        "sentence_text": normclaim
                    }}
                ]
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

def predictTop(query, sentence):
    prediction = predictor.predict(
        hypothesis=query,
        premise=sentence
    )
    p_pos = prediction['label_probs'][0]
    p_neg = prediction['label_probs'][1]
    p_neu = prediction['label_probs'][2]
    max_prob = max(p_pos, p_neg, p_neu)
    if max_prob == p_pos:
        return [max_prob,'SUPPORTS']
    elif max_prob == p_neg:
        return [max_prob,'REFUTES']
    else:
        return [max_prob,'NOT ENOUGH INFO']

#not lemmitized
def entityRetrieval2(query):
    results = nerPredicts.predict_json({"sentence": query})
    entity = []
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            entity.append(results['words'][index])
    return entity

def entityRetrieval3(query):
    results = nerPredicts.predict_json({"sentence": query})
    entity = []
    wordList=[]
    flag = False
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            # wordList.append(results['words'][index].lower())
            if str(tag).endswith("PER"):
                if str(tag).startswith('B-'):
                    phrase = results['words'][index]
                    flag = True
                elif flag:
                    phrase += " " + results['words'][index]
                    if str(tag).startswith('L-'):
                        flag = False
                        entity.append({ "match_phrase": {
                        "title":phrase}})
                else:
                    entity.append({ "match_phrase": {
                        "title": results['words'][index]}})
    return entity

if __name__ == '__main__':
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    inforPredictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/openie-model.2018-08-20.tar.gz")

    nerPredicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")

    with open('../Resource/devset100.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    fullResult = {}
    time1=time.time()
    for key, content in d.items():
        claim = content['claim']
        normClaim=" ".join([lemmatize(word.lower()) for word in claim.split(" ")])
        entities = entityRetrieval3(claim)
        evidenceList=[]
        sentence=""
        print(entities)
        infopredict = inforPredictor.predict_json({"sentence": claim})
        # print(predict)
        info = infopredict['verbs'][0]['description'].split("] [")
        np = ""
        for idx, item in enumerate(info):
            if item.startswith("V"):
                np = info[idx - 1].split(": ")[1]
        # print(np)
        if np not in entities:
            entities.append({"match_phrase": {
                "title": np}})
        # major vote
        # result = Counter()
        # for candidate in senSelection(url,claim,query)['hits']['hits']:
        #     evidenceList.append([candidate['_source']["page_identifier"], candidate['_source']["sentence_number"]])
        #     sentence = candidate['_source']["sentence_text"]
        #     result[predict(normClaim, sentence)] += 1
        # if len(result)>1:
        #     [(judge, count)] = result.most_common(1)
        # else:
        #     evidenceList=[]
        #     judge='NOT ENOUGH INFO'

        # #top one
        # for candidate in senSelection(url,claim,query)['hits']['hits']:
        #     if evidenceList == []:
        #         sentence = candidate['_source']["sentence_text"]
        #     evidenceList.append([candidate['_source']["page_identifier"],int(candidate['_source']["sentence_number"])])
        # judge = predict(normClaim, sentence)

        #top probability
        # pro=[]
        # for candidate in senSelection(url,claim,query)['hits']['hits']:
        #     sentence = candidate['_source']["sentence_text"]
        #     evidenceList.append([candidate['_source']["page_identifier"],int(candidate['_source']["sentence_number"])])
        #     pro.append(predictTop(normClaim, sentence))
        # if len(pro)>1:
        #     sorted(pro)
        #     judge=pro[0][1]
        # else:
        #     evidenceList=[]
        #     judge='NOT ENOUGH INFO'

        # logic
        judge=""
        for candidate in senSelection(url,normClaim,claim,entities)['hits']['hits']:
            if judge=="":
                sentence = candidate['_source']["sentence_text"]
                predicted = predict(normClaim, sentence)
                if predicted!='NOT ENOUGH INFO':
                    judge=predicted
                    # evidenceList = []
            evidenceList.append([candidate['_source']["page_identifier"],int(candidate['_source']["sentence_number"])])
        if judge=="":
            judge='NOT ENOUGH INFO'
        # if judge=='NOT ENOUGH INFO':
        #     evidenceList = []

        fresult = {}
        fresult['claim'] = content['claim']
        fresult['label'] = judge
        fresult['evidence'] = evidenceList
        fullResult[key] = fresult

    # store result
    with open('./logic6.json', 'w', encoding='utf-8') as f:
        json.dump(fullResult, f)
        f.close()
