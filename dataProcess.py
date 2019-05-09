import nltk
import os
from collections import Counter

def dataProcessing():
    # Save to the current directory
    files = os.listdir('./wiki-pages-text/')
    norm_docs = {}
    stemmer = nltk.stem.PorterStemmer()
    
    """
    #subtitude function
    lemmatizer = nltk.stem.wordnet.WordNetLemmatizer()
    def lemmatize(word):
        lemma = lemmatizer.lemmatize(word,'v')
        if lemma == word:
            lemma = lemmatizer.lemmatize(word,'n')
        return lemma
    """
    
    for f in files:
        with open('./wiki-pages-text/'+ f, 'r', encoding='utf-8') as f:
            for raw_doc in f:
                oneLine=raw_doc.split(" ")
                title=oneLine[0]
                senNum=oneLine[1]
                # stemming and lower
                raw_sentence=[stemmer.stem(word.lower()) for word in oneLine[2:] if word.isalpha()]
                cnt = Counter()
                doc_term_freqs=[]
                for word in raw_sentence:
                    cnt[word] += 1
                doc_term_freqs.append(cnt)
                if title not in norm_docs.keys():
                    norm_docs[title]=[{'senNum':senNum,'sentence':raw_sentence,'frequency':doc_term_freqs}]
                else:
                    norm_docs[title].append({'senNum':senNum,'sentence':raw_sentence,'frequency':doc_term_freqs})
    return norm_docs