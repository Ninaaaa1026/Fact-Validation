import json
import requests
from allennlp.predictors.predictor import Predictor
import nltk,time,os
import tensorflow as tf
import numpy as np
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

def senSelection(url, normclaim,entities):
#     """Simple Elasticsearch Query"""
#     # print(titles)
#     senquery=[]
    entityQuery = []
    for entity in entities:
        entityQuery.append({"match_phrase": {
            "title": entity}})
#         entityQuery.append({"match_phrase": {
#             "sentence_text": entity}})
#     entityQuery.append({"match": {
#                             "sentence_text": root
#                         }})
#     query = json.dumps({
#         "query": {
#             "bool": {
#                 # "must": senquery
#                 #  ,
#                 "should": entityQuery
#             }
#         },
#         "size": 5
#     })
    query = json.dumps({
        "query": {
            "bool": {
                "must": [
                    {"match": {
                        "sentence_text": normclaim
                    }}
                ]
                ,
                "should": entityQuery
            }
        },
        "size": 5
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    # print(results)
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
            # if str(tag).endswith("PER"):
            if str(tag).startswith('B-'):
                phrase = results['words'][index]
                flag = True
            elif flag:
                phrase += " " + results['words'][index]
                if str(tag).startswith('L-'):
                    flag = False
                    entity.append(phrase)
            else:
                entity.append(results['words'][index])
    return entity
class nnClassifier():
    def __init__(self):
        self.labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        self.sess = tf.InteractiveSession()
        self.build_network()
        self.saver = tf.train.Saver()


    def build_network(self):
        print("Building observer network...")

        self.xs = tf.placeholder(tf.float32, shape=(None,15))
        self.ys = tf.placeholder(tf.float32, [None, 3])

        hidden_layer = tf.layers.dense(
            inputs=self.xs,
            units=25,
            activation=tf.nn.relu
        )

        self.out_layer = tf.layers.dense(
            inputs=hidden_layer,
            units=3,
            activation=tf.nn.softmax
        )
        self.sess.run(tf.global_variables_initializer())
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.ys * tf.log(self.out_layer)))
        # loss = tf.reduce_mean(tf.reduce_sum(self.ys  * tf.log(self.out_layer)),reduction_indices=[1])
        self.train_step = tf.train.GradientDescentOptimizer(0.1).minimize(self.loss)

    def train(self, batch_xs, batch_ys):
        # input=np.array(batch_xs).transpose()
        # print(input)
        [_, loss_val] = self.sess.run([self.train_step, self.loss],feed_dict={self.xs:np.array(batch_xs)[np.newaxis, :],self.ys:np.array(batch_ys)[np.newaxis, :]})

        return loss_val

    def classify(self,inputEntailment):
        print(inputEntailment)
        output=self.sess.run(self.out_layer, feed_dict={self.xs: np.array(inputEntailment)[np.newaxis, :]})
        print(output)
        index=int(np.argmax(output))
        return self.labels[index]

    def restoreNet(self):
        if os.path.exists("NeuralNetwork"):
            self.saver.restore(self.sess, "NeuralNetwork/save_net.ckpt")

    def storeNet(self):
        save_path = self.saver.save(self.sess, "NeuralNetwork/save_net.ckpt")
        print("Save to path: ", save_path)

if __name__ == '__main__':
    nerPredicts = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/ner-model-2018.12.18.tar.gz")
    predictor = Predictor.from_path(
        "https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")

    inforPredictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/biaffine-dependency-parser-ptb-2018.08.23.tar.gz")

    with open('../Resource/devset100.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()
    labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    nn = nnClassifier()
    nn.restoreNet()

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
        info = " ".join(infoPredict['words'][:idx + 1])
        # root = lemmatize(infoPredict['hierplane_tree']['root']['word'].lower())
        tag=False
        if info not in entities:
            for entity in entities:
                if entity in info:
                    tag=True
                    break
        else:tag=True
        if not tag:
            entities.append(info)

        pro = []
        count = 0
        for candidate in senSelection(url, normClaim,entities)['hits']['hits']:
            count += 1
            pro.extend(predictor.predict(hypothesis=normClaim, premise=candidate['_source']["sentence_text"])[
                           'label_probs'])
            evidenceList.append(
                [candidate['_source']["page_identifier"], int(candidate['_source']["sentence_number"])])
        while count < 5:
            pro.extend([0, 0, 1])
            count += 1
        fresult = {}
        fresult['claim'] = content['claim']
        fresult['label'] = nn.classify(pro)
        fresult['evidence'] = evidenceList
        fullResult[key] = fresult

        # store result
    with open('./nnPredict.json', 'w', encoding='utf-8') as f:
        json.dump(fullResult, f)
        f.close()
