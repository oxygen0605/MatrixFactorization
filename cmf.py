# https://qiita.com/keitakurita/items/3a12708dc87e76497832

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

import pycmf

train = pd.read_csv("./python/MatrixFactorization/data/train.csv")

vectorizer = CountVectorizer(min_df=3, max_df=0.9, stop_words="english", binary=True)
X_train = vectorizer.fit_transform(train.head(10000).comment_text)

N_COMPONENTS = 20
# for CMF
label_columns = list(train.columns[2:])
Y_train = train.head(10000)[label_columns].values

# alpha(hyperparameter)の計算
xnorm = np.sqrt(X_train.multiply(X_train).sum())
ynorm = np.sqrt((Y_train * Y_train).sum())
alpha = (ynorm / (xnorm + ynorm)) ** 0.75

cmf = pycmf.CMF(N_COMPONENTS,
               U_non_negative=True, V_non_negative=True, Z_non_negative=False,
               x_link="linear", y_link="logit", alpha=alpha, l1_reg=2., l2_reg=5., max_iter=10,
               solver="newton", verbose=True)
U, V, Z = cmf.fit_transform(X_train.T, Y_train)

cmf.print_topic_terms(vectorizer)