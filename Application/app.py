import json
import logging
import re

import keras
import numpy as np
import tensorflow as tf
import unidecode
from bs4 import BeautifulSoup
from flask import Flask, request, jsonify
from flask_cors import CORS
from gevent.pywsgi import WSGIServer
import urllib.request


app = Flask(__name__)
CORS(app)

model = None
graph = None

with open('resources/config.json') as json_data_file:
    config = json.load(json_data_file)
vocab_to_int = config['vocab2int']
seq_length = config['max_len']

original_texts = {}


def clean_str(original_string):
    s = unidecode.unidecode(original_string)
    s = re.sub(r'\t', '', s)
    s = re.sub(r'\r', '', s)
    s = s.lower()
    s = re.sub(r'[^a-z0-9]', ' ', s)
    s = re.sub(r' +', ' ', s)
    result = s.strip()
    original_texts[result] = original_string
    return result


def pad_features(reviews_ints, this_seq_length):
    features = np.zeros((len(reviews_ints), this_seq_length), dtype=int)
    for i, row in enumerate(reviews_ints):
        features[i, :len(row)] = np.array(row)[:this_seq_length]
    return features


def spoil_me(data):
    if type(data) != "string":
        soup = BeautifulSoup(data)
        for script in soup(["script", "style", "meta"]):
            script.extract()
        text = soup.getText()
    else:
        text = data
    s = re.sub(r'\n', '', text).replace(". ", "")
    s = s.split('')
    s = [clean_str(i) for i in s]
    s = [f for f in s if f]
    s_ = [[vocab_to_int.get(j, 1) for j in k.split()] for k in s]
    s_ = pad_features(s_, seq_length)
    predictions = model.predict(s_)
    predictions_ = [p[1] for p in predictions]
#    predictions_cat = predictions.argmax(axis=1)
    result = list(zip(s, predictions_))
    spoilers = [r for r in result if r[1] >= 0.5]
    for i, val in enumerate(spoilers):
        spoiler = spoilers[i]
        spoilers[i] = (original_texts[spoiler[0]], spoiler[1])

    return text, list(enumerate(spoilers)), predictions


def build_model():
    built_model = keras.models.Sequential()
    built_model.add(keras.layers.Embedding(len(vocab_to_int), 16))
    built_model.add(keras.layers.GlobalAveragePooling1D())
    built_model.add(keras.layers.Dense(16, activation='relu'))
    built_model.add(keras.layers.Dense(2, activation='sigmoid'))
    return built_model


def load_model():
    global model
    global graph
    model = build_model()
    model.build(input_shape=(None, seq_length))
    model.load_weights('resources/model_weights.h5')
    graph = tf.get_default_graph()
    print('Model ready! Go to http://127.0.0.1:5000/')


def serialize_spoiler(spoiler):
    return str(spoiler[1][1]), spoiler[1][0]


@app.route('/', methods=['POST'])
def predict():
    if request.method == 'POST':
        with graph.as_default():
            if request.headers['Content-Type'] == 'text/html':
                original_text, spoilers, all_predictions = spoil_me(request.stream.read().decode("utf-8"))

            elif request.headers['Content-Type'] == 'application/json':
                args = request.json
                if 'html' in args:
                    html = request.json["html"]
                    original_text, spoilers, all_predictions = spoil_me(html)
                elif 'url' in args:
                    url = request.json["url"]
                    with urllib.request.urlopen(url) as response:
                        original_text, spoilers, all_predictions = spoil_me(response.read())
                else:
                    return jsonify({"error": "No 'html' nor 'url' specified"}), 400

            serialized_spoilers = list(map(serialize_spoiler, spoilers))

        return jsonify(serialized_spoilers)

    return "Send text!"


if __name__ == "__main__":
    load_model()
    http_server = WSGIServer(('0.0.0.0', 5000), app)
    http_server.serve_forever()
