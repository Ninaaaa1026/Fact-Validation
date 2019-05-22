import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.externals import joblib
import csv,json,requests
from allennlp.predictors.predictor import Predictor

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

#python2可以用file替代open
with open("trainRTE.csv","w") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["pro1","pro2","pro3","pro4","pro5","pro6","pro7","pro8","pro9","pro10","pro11","pro12","pro13","pro14","pro15","label"])

    predictor = Predictor.from_path("https://s3-us-west-2.amazonaws.com/allennlp/models/decomposable-attention-elmo-2018.02.19.tar.gz")
    with open('train.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
        f.close()

    labels=["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
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
            for doc in searchSentence(url, title, str(senNum))['hits']['hits']:
                premise =doc['_source'][ 'sentence_text']
            predict=predictor.predict(hypothesis=claim,premise=premise)
            pro.extend(predict['label_probs'])
            count+=1
            if count>=5:
                break
        while count<5:
            pro.extend(predictor.predict(hypothesis=claim,premise=""))
            count+=1

        for idx,lab in enumerate(labels):
            if lab==label:
                ys=idx
        pro.extend(ys)
        writer.writerow(pro)


data = pd.read_csv("./trainRTE.csv")
x_rows = []
for x in data.iterrows():
    if x not in ['label']:
        x_rows.append(x)
X = data[x_rows]
y = data['label']
x_train, x_test, y_train, y_test = train_test_split(X, y)

# 模型训练，使用GBDT算法
gbr = GradientBoostingClassifier(n_estimators=3000, max_depth=2, min_samples_split=2, learning_rate=0.1)
gbr.fit(x_train, y_train.ravel())
joblib.dump(gbr, 'train_model_result4.m')   # 保存模型

y_gbr = gbr.predict(x_train)
y_gbr1 = gbr.predict(x_test)
acc_train = gbr.score(x_train, y_train)
acc_test = gbr.score(x_test, y_test)
print(acc_train)
print(acc_test)

