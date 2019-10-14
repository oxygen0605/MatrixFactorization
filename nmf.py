# https://qiita.com/keitakurita/items/3a12708dc87e76497832

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

train = pd.read_csv("./python/MatrixFactorization/data/train.csv")

vectorizer = CountVectorizer(min_df=3, max_df=0.9, stop_words="english", binary=True)
X_train = vectorizer.fit_transform(train.head(10000).comment_text)

N_COMPONENTS = 20
from sklearn.decomposition import NMF
nmf = NMF(N_COMPONENTS)
W = nmf.fit_transform(X_train)
H = nmf.components_

idx_to_word = np.array(vectorizer.get_feature_names())
def print_topics(H, topn=10):
    for i, topic in enumerate(H): 
        print("Topic {}: {}".format(i + 1, ",".join([str(x) for x in idx_to_word[topic.argsort()[-topn:]]])))

print_topics(H)