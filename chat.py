import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import tflearn
from tensorflow.python.framework import ops
import random
import json
import pickle


stemmer = LancasterStemmer()
ERROR_THRESHOLD = 0.5
context = {}


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return np.array(bag)


def classify(sentence):
    results = model.predict([bow(sentence, words)])[0]
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))
        return return_list


def response(sentence):
    results = classify(sentence)
    if results:
        while results:
            for i in intents['intents']:
                if i['tag'] == results[0][0]:
                    # if random.choice(i['responses']) is None:
                    #     return random.choice(["Sorry, can't understand you", "Please give me more info",
                    #                           "Not sure I understand"])
                    # else:
                    return random.choice(i['responses'])
            results.pop(0)


with open('intents.json') as json_data:
    intents = json.load(json_data)
    data = pickle.load(open("training_data", "rb"))
    words = data['words']
    classes = data['classes']
    train_x = data['train_x']
    train_y = data['train_y']
    # ops.reset_default_graph()
    network = tflearn.input_data(shape=[None, len(train_x[0])])
    network = tflearn.fully_connected(network, 8)
    network = tflearn.fully_connected(network, 8)
    network = tflearn.fully_connected(network, len(train_y[0]), activation='softmax')
    network = tflearn.regression(network)
    model = tflearn.DNN(network)
    model.load(model_file='MyModel', weights_only=False)

    bot_name = 'Bot'
    print("Let's chat!")
    while True:
        sentence = input('You: ')
        ans = response(sentence)
        if ans is None:
            ans = random.choice(["Sorry, can't understand you", "Please give me more info", "Not sure I understand"])
        print(f'{bot_name}: {ans}')
        if ans in ["See you later", "Have a nice day", "Bye! Come back again soon."]:
            break
