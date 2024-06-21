import os
import json
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, render_template
import pickle
import re

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'lstm_model3k.h5')
model = load_model(model_path)

with open(os.path.join(os.path.dirname(__file__), 'tokenizer.pkl'), 'rb') as handle:
    tokenizer = pickle.load(handle)

nltk.download('stopwords')
stop_words = set(stopwords.words('arabic'))

MAX_SEQUENCE_LENGTH = 128

data = {
    'acc1': 88.54,
    'acc2': 86.25,
    'acc3': 85.96,
    'acc4': 84.49
}

def removeDiacretics(news_list):
    arabic_diacritics = re.compile("""
                             ّ    | # Tashdid
                             َ    | # Fatha
                             ً    | # Tanwin Fath
                             ُ    | # Damma
                             ٌ    | # Tanwin Damm
                             ِ    | # Kasra
                             ٍ    | # Tanwin Kasr
                             ْ    | # Sukun
                             ـ     # Tatwil/Kashida
                         """, re.VERBOSE)
    temp_list = list()
    for news in news_list:
        text = re.sub(arabic_diacritics, '', news)
        temp_list.append(text)
    return temp_list

def normalizeArabic(text):
    text = re.sub("[إأٱآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    return text

@app.route('/')
def index():
    return render_template('index.html', data=data)

@app.route('/synthesis', methods=['POST'])
def predict():
    user_input = request.form['txt']
    processed_input = ' '.join([word for word in user_input.split() if word not in stop_words])
    sequence = tokenizer.texts_to_sequences([processed_input])
    padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH, padding='post')
    prediction_probs = model.predict(padded_sequence)
    prediction = np.argmax(prediction_probs, axis=1)[0]
    labels = ['credible', 'not credible', 'undecided']
    result = labels[prediction]
    return render_template('index.html', result=result, data=data, user_input=user_input)

if __name__ == '__main__':
    app.run(debug=True)
