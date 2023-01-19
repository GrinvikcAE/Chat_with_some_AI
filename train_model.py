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

with open('intents.json') as json_data:
    intents = json.load(json_data)

words = []
classes = []
documents = []
words_for_ignore = ['?', '!']
for intent in intents['intents']:
    for pattern in intent['patterns']:
        word = nltk.word_tokenize(pattern)
        words.extend(word)
        documents.append((word, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
words = [stemmer.stem(word.lower()) for word in words if word not in words_for_ignore]
words = sorted(list(set(words)))
classes = sorted(list(set(classes)))
# print(f'{len(documents)} documents')
# print(f'{len(classes)} classes {classes}')
# print(f'{len(words)} uniq stemmed words {words}')

training = []
output = []
output_empty = [0] * len(classes)

for doc in documents:
    bag = []
    pattern_words = doc[0]
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    for word in words:
        if word in pattern_words:
            bag.append(1)
        else:
            bag.append(0)
    # print(f'{pattern_words} pattern_words')
    output_row = list(output_empty)
    # print(output_row)
    output_row[classes.index(doc[1])] = 1
    # print(output_row)
    training.append([bag, output_row])
    # print(training)

random.shuffle(training)
training = np.array(training)
# print(f'{training} array for training')
train_x = list(training[:, 0])
train_y = list(training[:, 1])
# print(train_x)
# print(train_y)

# tf.reset_default_graph()
ops.reset_default_graph()
network = tflearn.input_data(shape=[None, len(train_x[0])])
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, 8)
network = tflearn.fully_connected(network, len(train_y[0]), activation='softmax')
network = tflearn.regression(network)

model = tflearn.DNN(network, tensorboard_dir='tflearn_logs')
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('MyModel')
pickle.dump({'words': words, 'classes': classes, 'train_x': train_x, 'train_y': train_y}, open("training_data", "wb"))
