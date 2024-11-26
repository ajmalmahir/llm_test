## import the required libraries and APIs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

print(tf.__version__)

## load the imdb reviews dataset
data, info = tfds.load("imdb_reviews", with_info=True, as_supervised=True)

## segregate training and test set
train_data, test_data = data['train'], data['test']

## create empty list to store sentences and labels
train_sentences = []
test_sentences = []

train_labels = []
test_labels = []

## iterate over the train data to extract sentences and labels
for sent, label in train_data:
    train_sentences.append(str(sent.numpy().decode('utf8')))
    train_labels.append(label.numpy())

for sent, label in test_data:
    test_sentences.append(str(sent.numpy().decode('utf8')))
    test_labels.append(label.numpy())