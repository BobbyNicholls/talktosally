# -*- coding: utf-8 -*-
"""
Created on Sun May  3 22:20:02 2020

@author: Rob
"""

###
###
###

import pandas as pd
import numpy as np
import pickle
from keras import backend

from utils import vectorise_texts, get_embedding_matrix, make_word_level_model

###
###
###

CELLS = 64
SAVE_MODEL = True
EPOCHS = 50
#training_ratio = 0.9
STRIDE = 25
BATCH_SIZE = 2048
VERBOSE = 1
model_name = 'test2'
model_dir = '' #lleave empty to just save in root dir of project


max_features = 5000
maxlen = 400
batch_size = 2
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

questions_df = pd.read_csv("question_data.csv")

###
###
###

text_col = []
label_col = []
for i, col in enumerate(questions_df.columns):
    text_col += list(questions_df[col])
    label_col += [i]*len(questions_df)

master_df  = pd.DataFrame({'text' : text_col, 
                           'label' : label_col})
    
from sklearn.utils import shuffle
master_df = shuffle(master_df)

vector_df, word_idx, tokenizer = vectorise_texts(master_df, col = 'text')
#pickle.dump(tokenizer, open('tokenizer.pickle','wb'))
embedding_matrix = get_embedding_matrix(word_idx)

def pad_vectors(vect, vect_len = 10, col = 'word_sequences'):
    """
    add trailing 0s to our vectors until they are of the length in the args
    """
    if len(vect) > 10:
        return vect[:10]
    else:
        return vect+[0]*(10-len(vect))

master_df['padded_vecs'] = [pad_vectors(x) for x in master_df['word_sequences'].to_numpy()]

features = np.array([np.array(x) for x in master_df['padded_vecs']])

print("so our text goes in to the bot looking like this:")
print(features)

print("with each of those numbers mapping to a vector like this:")
print(embedding_matrix[1,:])

labels = master_df['label'].to_numpy()



model = make_word_level_model(
#    num_words,
    embedding_matrix=embedding_matrix,
#    lstm_cells=LSTM_CELLS,
#    trainable=True,
#    lstm_layers=1
    )

from sklearn.preprocessing import OneHotEncoder
onehot_encoder = OneHotEncoder(sparse=False)
labels = onehot_encoder.fit_transform(labels.reshape(-1,1))

model.fit(features, 
          labels,
          batch_size=2,
          epochs=20,
#          validation_data=(x_test, y_test)
          )

#model.save('first_model.hdf5')

#backend.clear_session()
input_seq = tokenizer.texts_to_sequences(['can you tell me a story?'])[0]

if len(input_seq) > 10:
    input_seq =  vect[:10]
else:
    input_seq = input_seq+[0]*(10-len(input_seq))

pred = model.predict(np.array(input_seq).reshape(1,-1))

import matplotlib.pyplot as plt

categories = ['general news request', 
              'greeting',
              'species information request',
              'request for story']

plt.bar(categories,pred[0])

print(f"Sally thinks you want to know about: {categories[np.argmax(pred)]}")


