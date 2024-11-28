## import the required libraries and APIs
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import keras
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

# print(tf.__version__)

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

## iterate over teh test set to extract sentences and labels
for sent, label in test_data:
    test_sentences.append(str(sent.numpy().decode('utf8')))
    test_labels.append(label.numpy())

## convert lists into numpy array
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

## define the parameters for the tokenizing and padding
vocab_size = 10000
embedding_dim = 16
max_length = 150
trunc_type = "post"
oov_tok = "<oov>"

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

## training sequences and labels
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_padded = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type)

## testing sequences and labels 
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_padded = pad_sequences(test_seq, maxlen=max_length)

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

# print(train_sentences[1])
# print(train_padded[1])
# print(decode_review(train_padded[1]))

## define the Neural Network with Embedding Layer
## 1. use a sequential api
## 2. add an embedding input layer of input size equal to vocabulary size
## 3. add a flatten layer, and two dense layers.

model = keras.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Flatten(),
    keras.layers.Dense(6, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

print(train_padded.shape)
print(test_padded.shape)

## compile the model with loss function, optimiser and metrics 
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# model.summary()