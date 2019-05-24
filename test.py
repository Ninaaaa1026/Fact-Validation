import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk,time
from collections import Counter
import re
url = "http://localhost:9200/collections/_search"
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
lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, 'a')
    return lemma

if __name__ == '__main__':
    strSearch="Hedda Gabler's world premiere took place at the Battle of Hastings."

    claim=" ".join([lemmatize(word.lower()) for word in strSearch.split(" ")])
    dependPredictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

    nerPredicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    entities=entityRetrieval3(strSearch)
    print(entities)
    # constituencyPredictor = Predictor.from_path(
    #     "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo-constituency-parser-2018.03.14.tar.gz")

    predict = dependPredictor.predict_json({"sentence": strSearch})
    print(predict)
    idx=0
    if 'nsubjpass'in predict['predicted_dependencies'] :
        idx=predict['predicted_dependencies'].index('nsubjpass')
    elif 'nsubj'in predict['predicted_dependencies']:
        idx = predict['predicted_dependencies'].index('nsubj')
    np=" ".join(predict['words'][:idx+1])
    root=lemmatize(predict['hierplane_tree']['root']['word'].lower())
    # constituencyPredictor.predict_json({"sentence":claim})
    print(root)
    # info = predict['verbs'][0]['description'].split("] [")
    # np = ""
    # for idx, item in enumerate(info):
    #     if item.startswith("V"):
    #         np= info[idx - 1].split(": ")[1]
    print(np)
    if np not in entities:
        entities.append({ "match_phrase": {
                        "title": np}})
    print(entities)
    query = json.dumps({
        "query": {
            "bool": {
                "must": [
                    { "match": {
                        "sentence_text": claim
                    }}
                ]
                ,
                "should":entities
            }
        },
        "size": 5
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    print(results)

