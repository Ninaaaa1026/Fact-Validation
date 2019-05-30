import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk
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

def senSelection(url, normclaim,root,entities):
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
                    "query": normclaim,
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
    try:
        print(query)
        results = nerPredicts.predict_json({"sentence": query})
    except RuntimeError:
        return []
    else:
        entity = []
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
    fineNerPredicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/fine-grained-ner-model-elmo-2018.12.21.tar.gz")
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    # srlPredictor = Predictor.from_path(
    #     "https://s3-us-west-2.amazonaws.com/allennlp/models/srl-model-2018.05.25.tar.gz")

    inforPredictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")
    # with open('./devset500.json', 'r', encoding='utf-8') as f:
    with open('./test-unlabelled.json', 'r', encoding='utf-8') as f:
    # with open('../Resource/test.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    fullResult = {}
    for key, content in d.items():
        claim = content['claim']
        normClaim=" ".join([lemmatize(word.lower()) for word in claim.split(" ")])
        entities = entityRetrieval3( claim)
        evidenceList=[]
        sentence=""
        infoPredict = inforPredictor.predict_json({"sentence": claim})
        idx = 0
        if 'nsubjpass' in infoPredict['predicted_dependencies']:
            idx = infoPredict['predicted_dependencies'].index('nsubjpass')
        elif 'nsubj' in infoPredict['predicted_dependencies']:
            idx = infoPredict['predicted_dependencies'].index('nsubj')
        infor = " ".join(infoPredict['words'][:idx + 1])
        root = lemmatize(infoPredict['hierplane_tree']['root']['word'].lower())
        tag=False
        if infor not in entities:
            for entity in entities:
                if entity in infor:
                    tag=True
                    break
        else:tag=True
        if not tag:
            entities.append(infor)

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
        judge=''
        for candidate in senSelection(url,normClaim,root,entities)['hits']['hits']:
            a = re.sub("-LRB-[\s\w]+-RRB-", "", candidate['_source']["title"])
            # print(a)
            resultNER = entityRetrieval3(a)
            if resultNER == []:
                resultNER = entityRetrieval4(a)
                if resultNER == []:
                    resultNER = [a.strip(" ")]
            pattern = re.search("-LRB-[\s\w]+-RRB-", candidate['_source']["title"])
            if pattern:
                inculded = pattern.group().replace("-LRB-", "").replace(
                    "-RRB-", "")
                resultNER.extend(entityRetrieval3(inculded))

            flag = False
            for ner in resultNER:
                if ner.replace("'s", "") in entities:
                    flag = True
                    break
            if flag and len(evidenceList)<1:
                sentence = candidate['_source']["sentence_text"]
                label = predict(normClaim, sentence)
                if evidenceList == []:
                    judge = label
                if judge==label:
                    evidenceList.append([candidate['_source']["page_identifier"],int(candidate['_source']["sentence_number"])
                                 ])
            if len(evidenceList) >= 1:
                break
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
        # judge=""
        # for candidate in senSelection(url,normClaim,root,entities)['hits']['hits']:
        #     # print(candidate['_source']["title"])
        #     a = re.sub("-LRB-[\s\w]+-RRB-", "", candidate['_source']["title"])
        #     # print(a)
        #     resultNER = entityRetrieval3(a)
        #     if resultNER == []:
        #         resultNER = entityRetrieval4(a)
        #         if resultNER == []:
        #             resultNER = [a.strip(" ")]
        #     pattern = re.search("-LRB-[\s\w]+-RRB-", candidate['_source']["title"])
        #     if pattern:
        #         inculded = pattern.group().replace("-LRB-", "").replace(
        #             "-RRB-", "")
        #         resultNER.extend(entityRetrieval3(inculded))
        #     # print("aaa" + str(resultNER))
        #     # print("bbb" + str(entities))
        #     flag = False
        #     for ner in resultNER:
        #         if ner.replace("'s","") in entities:
        #             # print("aaaa"+str(ner))
        #             flag = True
        #             break
        #     if flag and len(evidenceList) < 1:
        #         sentence = candidate['_source']["sentence_text"]
        #         label = predict(normClaim, sentence)
        #         # print(label)
        #         if judge == "":
        #             if label != 'NOT ENOUGH INFO':
        #                 judge = label
        #                 # evidenceList = []
        #         # if judge == label:
        #         evidenceList.append(
        #                 [candidate['_source']["page_identifier"], int(candidate['_source']["sentence_number"])])
        #     if len(evidenceList) >= 1:
        #         break
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
    with open('./testoutput.json', 'w', encoding='utf-8') as f:
    # with open('./json9.json', 'w', encoding='utf-8') as f:
        json.dump(fullResult, f)
        f.close()
