from keras._tf_keras.keras.preprocessing.text import Tokenizer

train_sentences = [
    'It is a sunny day',
    'It is a cloudy day',
    'Will it rain today?'
]

tokenizer = Tokenizer(num_words=100)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index

## create sequences using tokenizer
sequences = tokenizer.texts_to_sequences(train_sentences)

## print word index dictionary and sequences
print(f"word index ->{word_index}")
print(f"sequence of words ->{sequences}")

new_sentences = {
    'Will it be raining today?',
    'It is a pleasant day.'
}

new_sequences = tokenizer.texts_to_sequences(new_sentences)

## won't recognise new words, will only print words that have been encoded/trained
print(new_sentences)
print(new_sequences)

## set up the tokenizer again with the oov_token (out of vocabulary token)
tokenizer = Tokenizer(num_words=100, oov_token="<oov>")

## train the new tokenizer on training sentences
tokenizer.fit_on_texts(train_sentences)

## store word index for the words in the sentences
word_index = tokenizer.word_index

## create sequences of the new sentences
new_sequences = tokenizer.texts_to_sequences(new_sentences)
print(word_index)
print(new_sequences)
