## introduction to word encodings ##

## import the tensorflow APIs
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing.text import Tokenizer


## sentences to tokenize
train_sentences = [
    'It is a sunny day',
    'It is a cloudy day'
]

## instantiate the tokenizer
tokenizer = Tokenizer(num_words=100)

## train the tokenizer on training sentences
tokenizer.fit_on_texts(train_sentences)

## store word index for the words in the sentence
word_index = tokenizer.word_index

print(word_index)
