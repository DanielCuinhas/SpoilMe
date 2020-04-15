from os import environ
from pydoc import html
from urllib.parse import parse_qs

import pandas as pd
import flask
from flask import Flask, request, jsonify, render_template
import pickle
from gevent.pywsgi import WSGIServer
import json
import keras
import tensorflow as tf
from keras.models import model_from_json
from pprint import pprint
import numpy as np
import re
import unidecode


app = Flask(__name__)
model = None
graph = None

with open('resources/config.json') as json_data_file:
    config = json.load(json_data_file)
vocab_to_int=config['vocab2int']
seq_length=config['max_len']

def clean_str(s):
    s = unidecode.unidecode(s)
    s = re.sub(r'\t','',s)
    s = re.sub(r'\r','',s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9]',' ',s)
    s = re.sub(r' +',' ',s)
    return s.strip()


def pad_features(reviews_ints, seq_length):
    features = np.zeros((len(reviews_ints), seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, :len(row)] = np.array(row)[:seq_length]
    return features


def spoilme(text):
    s = re.sub(r'\n','.',text)
    s = s.split('.')
    s = [clean_str(i) for i in s]
    s = [f for f in s if f]
    s_ = [[vocab_to_int.get(j,1) for j in k.split()] for k in s]
    s_ = pad_features(s_,seq_length)
    predictions = model.predict(s_)
    predictions_ = [p[1] for p in predictions]
    predictions_cat = predictions.argmax(axis=1)
    result = list(zip(s, predictions_))
    spoilers = [r for r in result if r[1]>=0.5]
    return text, list(enumerate(spoilers)), predictions


def build_model():
    model = keras.models.Sequential()
    model.add(keras.layers.Embedding(len(vocab_to_int), 16))
    model.add(keras.layers.GlobalAveragePooling1D())
    model.add(keras.layers.Dense(16, activation='relu'))
    model.add(keras.layers.Dense(2, activation='sigmoid'))
    return model


def load_model():
    global model
    global graph
    global sess
    model=build_model()
    model.build(input_shape=(None, seq_length))
    model.load_weights('resources/model_weights.h5')
    graph = tf.get_default_graph()
    print('Model ready! Go to http://127.0.0.1:5000/')

@app.route('/')
def home():
    return render_template('index.html')

def serializeSpoiler(spoiler):
    return str(spoiler[1][1]), spoiler[1][0]

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        with graph.as_default():
            original_text, spoilers, all_predictions = spoilme(request.json["text"])
            serializedSpoilers = list(map(serializeSpoiler, spoilers))

        return jsonify(serializedSpoilers)

    return "Send 'text'!"
    
if __name__ == "__main__":
    load_model()
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
