import tensorflow as tf
import numpy as np
import os,time
import json
from whoosh.index import open_dir
from allennlp.predictors.predictor import Predictor
from whoosh.fields import *
from whoosh.qparser import QueryParser

class nnClassifier():
    def __init__(self):
        self.labels = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
        # self.real_result = []
        # self.index = 0
        self.sess = tf.InteractiveSession()
        self.build_network()
        self.saver = tf.train.Saver()

    def build_network(self):
        print("Building observer network...")

        self.xs = tf.placeholder(tf.float32, [None,3])
        self.ys = tf.placeholder(tf.float32, [None, 3])

        hidden_layer = tf.layers.dense(
            inputs=self.xs,
            units=100,
            activation=tf.nn.sigmoid
        )

        self.out_layer = tf.layers.dense(
            inputs=hidden_layer,
            units=3,
            activation=tf.nn.softmax
        )
        self.sess.run(tf.global_variables_initializer())
        loss = tf.reduce_mean(tf.reduce_sum(self.ys * tf.log(self.out_layer)))
        # loss = tf.reduce_mean(tf.reduce_sum(self.ys  * tf.log(self.out_layer)),reduction_indices=[1])
        self.train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

    def train(self, batch_xs, batch_ys):
        self.sess.run(self.train_step, feed_dict={self.xs: batch_xs,self.ys:batch_ys})

    def classify(self,inputEntailment):
        index=int(np.argmax(self.sess.run(self.out_layer, feed_dict={self.xs: inputEntailment})))
        return self.labels[index]

    def restoreNet(self):
        if os.path.exists("NeuralNetwork\my_net"):
            self.saver.restore(self.sess, "NeuralNetwork\my_net\save_net.ckpt")

    def storeNet(self):
        save_path = self.saver.save(self.sess, "NeuralNetwork\my_net\save_net.ckpt")
        print("Save to path: ", save_path)

if __name__ == '__main__':
    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    with open('./test.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    ix = open_dir("indexer")
    labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
    nn=nnClassifier()
      # results = searcher.document()
    with ix.searcher() as searcher:
        for key, content in d.items():
            claim = content['claim']
            evidence = content["evidence"]
            label=content["label"]
            premise=""
            pro=[]
            count=0
            for sentence in evidence:
                # count+=1
                title=sentence[0]
                senNum=sentence[1]
                print(title,senNum)
                time1=time.time()
                # premise=norm_docs[title][str(senNum)]
                query = QueryParser("content", ix.schema).parse(title)
                results = searcher.search(query)
                for r in results:
                    if r["path"]==str(senNum):
                        premise=r["content"]

                # myquery =
                # results = searcher.search(And([Term("path", senNum)], [Term("title",title)]))
                # print(searcher.document(title=u""+title))
                # print(searcher.stored_fields(searcher.document_number(title=u"" +title,path=u"" +str(senNum) )))
                # premise += searcher.documents(title=u"" + title, path=u"" + str(senNum))["content"]
                print(time.time()-time1)
                print(premise)
                predict=predictor.predict(hypothesis=claim,premise=premise)
                pro.append(predict['label_probs'])
            #     if count>7:
            #         break
            # while count<7:
            #     pro.append(predictor.predict(hypothesis=claim,premise="")['label_probs'])
            print(pro)
            ys=[]
            for idx,lab in enumerate(labels):
                if lab==label:
                    ys.append(1)
                else:
                    ys.append(0)
            nn.train(pro,ys)
    nn.storeNet()