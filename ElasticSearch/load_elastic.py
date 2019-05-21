# coding: utf-8

import os
import json
import requests
import numpy as np
import nltk

# Path configure
ROOT_PATH = "E:\教材\Semster3\Web Search\Assignment\project\Fact validation"
RESOURCE_PATH = os.path.join(ROOT_PATH, "Resource")
wiki_pages_text_path = os.path.join(RESOURCE_PATH, "wiki-pages-text")
train_json_path = os.path.join(RESOURCE_PATH, "train.json")
url = "http://localhost:9200/_bulk"

lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()

def lemmatize(word):
    lemma = lemmatizer.lemmatize(word, 'v')
    if lemma == word:
        lemma = lemmatizer.lemmatize(word, 'n')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word, 'a')
    return lemma

def load_data(temp_list):
    payload = ""
    for sentence in temp_list:
        page_identifier = sentence[0]
        sentence_number = sentence[1]
        sentence_text = sentence[2]
        fact_id = sentence[3]

        create_query = {"create": {"_index": "collections", "_type": "facts", "_id": fact_id}}
        data_query = {
            "page_identifier": page_identifier,
            "sentence_number": sentence_number,
            "page_title": " ".join([word.lower() for word in page_identifier.replace("_"," ")]),
            "sentence_text":  sentence_text,
        }
        encode_payload = json.dumps(create_query) + "\n" + json.dumps(data_query) + "\n"
        payload += encode_payload

    headers = {'Content-Type': "application/json",}

    response = requests.request("POST", url, data=payload, headers=headers)

    # print(response.text)


# Process wiki-pages-text data
def process_wiki_data():
    fact_id = 0
    for fileName in os.listdir(wiki_pages_text_path):
        wiki_document = os.path.join(wiki_pages_text_path, fileName)
        with open(wiki_document, "r", encoding="utf-8") as wikiFd:
            sentence_list = wikiFd.readlines()
            file_length = len(sentence_list)
            temp_sentence_list = list()

            for index, wikiSentence in enumerate(sentence_list):
                split_wiki_sentence = wikiSentence.split(" ")

                page_identifier = split_wiki_sentence[0]
                sentence_number = split_wiki_sentence[1]
                sentence_text = " ".join([lemmatize(word.lower())for word in split_wiki_sentence[2:]]).strip("\n")
                fact_id += 1

                temp_sentence_list += [(page_identifier, sentence_number, sentence_text, fact_id)]

                # Bulk operation
                if index % 15000 == 0:
                    load_data(temp_sentence_list)
                    temp_sentence_list = list()
                    print("Process: ", "%.2f" % (index / file_length * 100), "%")

            # Load remaining data
            load_data(temp_sentence_list)

            print("Done: ", fileName)


if __name__ == "__main__":
    process_wiki_data()