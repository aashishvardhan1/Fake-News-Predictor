# Imports
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from lightgbm import LGBMClassifier
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import pickle
import re





df= pd.read_csv('Data/train.csv')

#dropping missing values
df.dropna(subset=['title'], inplace=True)

#pre-processing
ps = PorterStemmer()
corpus = []
for text in df['title']:
    review = re.sub('[^a-zA-Z]', ' ', str(text))
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)



#using tokenizer
tokenizer = Tokenizer(num_words=50000)
tokenizer.fit_on_texts(corpus)
tkn_text = tokenizer.texts_to_matrix(corpus, mode='tfidf')

#Saving the model
pickle.dump(tokenizer, open('Models/tokenizer', 'wb'))


#model
Lgbm= LGBMClassifier()
Lgbm.fit(tkn_text, df['label'])


#Saving the model
pickle.dump(Lgbm, open('Models/LGBMClassifier_model', 'wb'))