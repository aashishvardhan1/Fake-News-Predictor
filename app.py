# Imports
import numpy as np
from flask import Flask, request, render_template
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import pickle
import re


app= Flask(__name__)
model_tokenizer= pickle.load(open('Models/tokenizer', 'rb')) 
model_predictor= pickle.load(open('Models/LGBMClassifier_model', 'rb')) 

ps = PorterStemmer()



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''

    text= request.form['text']
    
    #pre-processing
    txt = re.sub('[^a-zA-Z]', ' ', str(text))
    txt = txt.lower()
    txt = txt.split()
    txt = [ps.stem(word) for word in txt if not word in stopwords.words('english')]
    txt = ' '.join(txt)

    #tokenizing
    text_enc= model_tokenizer.texts_to_matrix([txt], mode='tfidf')
    prediction= model_predictor.predict(text_enc)
    

    return render_template('index.html', prediction_text= f'{np.where(prediction==1,"Fake News", "Real News")[0]}')

if __name__ == "__main__":
    app.run(debug=True)

