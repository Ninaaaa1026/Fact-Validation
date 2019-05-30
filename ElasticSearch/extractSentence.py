import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk,time
from collections import Counter
import re
url1 = "http://localhost:9200/_xpack/sql/"
url2 = "http://localhost:9200/collections/_search"
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, 'a')
    return lemma

def senSelection(url,doc,senid):
    query = json.dumps({
        "query": "SELECT sentence_text FROM collections WHERE page_identifier = '"+doc+"' AND sentence_number ="+str(senid)
    })
    response = requests.post(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    # print(results)
    return results

def senSelection2(url, normclaim,root,entities):
    entityQuery = []

    for entity in entities:
        entityQuery.append({"match_phrase": {
            "title": {"query": entity,
                    "boost": 4}}})

    if root not in entities:
        entityQuery.append(
            {
                "match_phrase": {
                    "sentence_text": {
                        "query": root,
                        "boost": 1
                    }
                }
            })
    # print(entityQuery)
    query = json.dumps({
        "query": {
            "bool": {
                "must": {"match": {
                "sentence_text": {
                    "query": normclaim
                    ,
                    "boost": 2
                }
            }}
                ,
                "should": entityQuery
            }
        },
        "size": 10
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    # print(results)
    return results

def entityRetrieval3(query):
    try:
        print(query)
        results = nerPredicts.predict_json({"sentence": query})
    except RuntimeError:
        return []
    else:
        entity = []
        wordList=[]
        flag = False
        for index, tag in enumerate(results['tags']):
            if len(str(tag)) > 1:
                if str(tag).startswith('B-'):
                    phrase = results['words'][index].strip(" ")
                    flag = True
                elif flag:
                    phrase += " " + results['words'][index].strip(" ")
                    if str(tag).startswith('L-'):
                        flag = False
                        entity.append(phrase.strip(" "))
                else:
                    entity.append(results['words'][index].strip(" "))
        return entity

def entityRetrieval4(query):
    results = fineNerPredicts.predict_json({"sentence": query})
    entity = []
    wordList=[]
    flag = False
    for index, tag in enumerate(results['tags']):
        if len(str(tag)) > 1:
            if str(tag).startswith('B-'):
                phrase = results['words'][index].strip(" ")
                flag = True
            elif flag:
                phrase += " " + results['words'][index].strip(" ")
                if str(tag).startswith('L-'):
                    flag = False
                    entity.append(phrase.strip(" "))
            else:
                entity.append(results['words'][index].strip(" "))
    return entity

if __name__ == '__main__':
    nerPredicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    inforPredictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    with open('./train.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()


    fullResult = {}
    for key, content in d.items():
        claim = content['claim']
        print(claim)
        label=content['label']
        if label!='NOT ENOUGH INFO':
            evidence = content["evidence"]
            evidenceList=[]
            for evi in evidence:
                try:
                    sen=senSelection(url1, evi[0], evi[1])['rows']
                    if sen !=[]:
                        evidenceList.append([evi[0],int(evi[1]),sen[0][0]])
                        print(sen[0][0])
                except:
                    continue
        else:
            fineNerPredicts = Predictor.from_path(
                "https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
            normClaim = " ".join([lemmatize(word.lower()) for word in claim.split(" ")])
            entities = entityRetrieval3(claim)
            evidenceList = []
            sentence = ""
            infoPredict = inforPredictor.predict_json({"sentence": claim})
            idx = 0
            if 'nsubjpass' in infoPredict['predicted_dependencies']:
                idx = infoPredict['predicted_dependencies'].index('nsubjpass')
            elif 'nsubj' in infoPredict['predicted_dependencies']:
                idx = infoPredict['predicted_dependencies'].index('nsubj')
            infor = " ".join(infoPredict['words'][:idx + 1])
            root = lemmatize(infoPredict['hierplane_tree']['root']['word'].lower())
            tag = False
            if infor not in entities:
                for entity in entities:
                    if entity in infor:
                        tag = True
                        break
            else:
                tag = True
            if not tag:
                entities.append(infor)
            for candidate in senSelection2(url2, normClaim,root,entities)['hits']['hits']:
                evidenceList.append(
                    [candidate['_source']["page_identifier"], int(candidate['_source']["sentence_number"])
                        , candidate['_source']["sentence_text"]
                     ])

        fresult = {}
        fresult['claim'] = content['claim']
        fresult['label'] = content["label"]
        fresult['evidence'] = evidenceList
        fullResult[key] = fresult

    # store result
    with open('./trainsenner.json', 'w', encoding='utf-8') as f:
        json.dump(fullResult, f)
        f.close()
