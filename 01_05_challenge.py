import pandas as pd
from keras._tf_keras.keras.preprocessing.text import Tokenizer
from keras._tf_keras.keras.preprocessing.sequence import pad_sequences

## read the data using the pandas library
data = pd.read_json("./x1.json")
data.head()

## create lists to store the headlines and labels
headlines = list(data['headline'])
labels = list(data['is_sarcastic'])

## set up the tokenizer 
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")
tokenizer.fit_on_texts(headlines)

word_index = tokenizer.word_index
print(word_index)

## create sequences of the headlines
sequences = tokenizer.texts_to_sequences(headlines)

## post-pad sequences
padded_seqs = pad_sequences(sequences, padding="post")

## printing padded sequences sample
print(padded_seqs[0])