# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:16:39 2020

@author: Rob
"""

#import json
from keras.preprocessing.text import Tokenizer
import pandas as pd
import numpy as np
from keras.layers import Embedding, Masking, LSTM, GRU, Bidirectional, Dense, Dropout, concatenate, Input, Activation
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras.models import Sequential, load_model, Model
from keras.optimizers import Adam
from keras.utils import plot_model
from keras.callbacks import EarlyStopping, ModelCheckpoint, BaseLogger
#from google.cloud import bigquery
from sklearn.preprocessing import MinMaxScaler
#import os
#import re

def get_embedding_matrix(word_idx, embedding_file = r'C:\Users\Rob\Desktop\Quandl\RNN\data\word_vectors\glove.6B.100d.txt'):

    glove = np.loadtxt(embedding_file, dtype='str', comments=None, encoding = 'utf-8')
    glove.shape
    # separate into the words and the vectors
    vectors = glove[:, 1:].astype('float')
    words = glove[:, 0]

    # Next we want to keep only those words that appear in our vocabulary. For words that are in our vocabulary but don't have an embedding, they will be represented as all 0s (a shortcoming that we can address by training our own embeddings.)

    word_lookup = {word: vector for word, vector in zip(words, vectors)}
    num_words = len(word_idx) + 1
    embedding_matrix = np.zeros((num_words, vectors.shape[1]))
    not_found = 0
    for i, word in enumerate(word_idx.keys()):
        # Look up the word embedding
        vector = word_lookup.get(word, None)

        # Record in matrix
        if vector is not None:
            embedding_matrix[i + 1, :] = vector
        else:
            not_found += 1

    print(f'There were {not_found} words without pre-trained embeddings.')

    return embedding_matrix

def vectorise_texts(df, col = 'text', filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'):
    # we should be doing some more pre-processing here for sure
    texts = list(df[col])
#    tokenizer = Tokenizer(filters=filters)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    word_sequences = tokenizer.texts_to_sequences(texts)
    df['word_sequences'] = word_sequences
    
    word_idx = tokenizer.word_index
    
    return df, word_idx, tokenizer


def make_word_level_model(
#                          num_words,
                          embedding_matrix,
#                          lstm_cells=64,
                          trainable=False,
                          lstm_layers=1,
                          #rob: bidirectional layer should be used if you want to thing to learn from the entire sequence every time, so it doesnt always look back, this is a major hyperparameter in our model since presumably we should train on only being able to see histroy, but bidirectional might help too
                          bi_direc=False,
                          embedding_dims=100,
                          maxlen=10,
#                          maxlen=1,
                          max_features=5000,
                          filters=100,
                          kernel_size=2,
                          hidden_dims=50
                          ):
    
    model = Sequential()
    
    # we start off with an efficient embedding layer which maps
    # our vocab indices into embedding_dims dimensions
    model.add(Embedding(max_features,
                        embedding_dims,
                        input_length=maxlen,
                        name='embedding'))
    
    model.add(Dropout(0.2))
    
    # we add a Convolution1D, which will learn filters
    # word group filters of size filter_length:
    model.add(Conv1D(filters,
                     kernel_size,
                     padding='valid',
                     activation='relu',
                     strides=1))
    
    # we use max pooling:
    model.add(GlobalMaxPooling1D())
    
    # We add a vanilla hidden layer:
    model.add(Dense(10, activation='relu'))
    model.add(Dropout(0.2))
#    model.add(Activation('relu'))
    
#    model.add(Dense(3))
#    model.add(Activation('softmax', name = 'act2'))
    model.add(Dense(4, activation='softmax', name = 'out'))
    
    model.compile(loss='categorical_crossentropy', 
                  optimizer='adam', 
                  metrics=['accuracy'])
    
    return model
    








