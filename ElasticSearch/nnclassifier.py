import tensorflow as tf
import numpy as np
import os,time
from allennlp.predictors.predictor import Predictor
import json
import requests
from elasticsearch import Elasticsearch

sumTotal=0
episode=0
url = "http://localhost:9200/_search"

def searchSentence(url, title, senNum):
    """Simple Elasticsearch Query"""
    query = json.dumps({
        "query": {
            "bool": {
                "must": [
                    {"match": {"page_identifier": title}},
                    {"match": {"sentence_number":senNum}}
                ]
            }
        }
    })
    response = requests.get(url, data=query,headers={'Content-Type': "application/json",})
    results = json.loads(response.text)
    return results

# def format_results(results):
#     """Print results nicely:  doc_id) content
#     """
#     data = [doc for doc in results['hits']['hits']]
#     for doc in data:
#         print("%s) %s" % (doc['_id'], doc['_source']['content']))


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
        index=int(np.argmax(self.sess.run(self.out_layer, feed_dict={self.xs: np.array(inputEntailment)[np.newaxis, :]})))
        return self.labels[index]

    def restoreNet(self):
        if os.path.exists("NeuralNetwork"):
            self.saver.restore(self.sess, "NeuralNetwork/save_net.ckpt")

    def storeNet(self):
        save_path = self.saver.save(self.sess, "NeuralNetwork/save_net.ckpt")
        print("Save to path: ", save_path)

if __name__ == '__main__':
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    with open('train.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    nn=nnClassifier()
    nn.restoreNet()
    for key, content in d.items():
        claim = content['claim']
        evidence = content["evidence"]
        label=content["label"]
        premise=""
        pro=[]
        count=0
        for sentence in evidence:
            title=sentence[0]
            senNum=sentence[1]
            print(title,senNum)
            # time1=time.time()
            for doc in searchSentence(url, title, str(senNum))['hits']['hits']:
                premise =doc['_source'][ 'sentence_text']
            # print(time.time()-time1)
            # print(premise)
            predict=predictor.predict(hypothesis=claim,premise=premise)
            pro.extend(predict['label_probs'])
            count+=1
            if count>=5:
                break
        while count<5:
            # pro.extend(predictor.predict(hypothesis=claim,premise="")['label_probs'])
            pro.extend([0,0,1])
            count+=1
        # if count==0:
        #     pro.append(predictor.predict(hypothesis=claim,premise="")['label_probs'])
        # print(pro)
        ys=[]
        for idx,lab in enumerate(labels):
            if lab==label:
                ys.append(1)
            else:
                ys.append(0)
        # print(ys)
        loss=nn.train(pro,ys)
        episode+=1
        sumTotal+=loss
        if episode%50==0:
            average=sumTotal/episode
            nn.storeNet()
            print(average)

    nn.storeNet()