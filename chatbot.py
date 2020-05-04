# -*- coding: utf-8 -*-
"""
Created on Mon May  4 10:42:46 2020

@author: Rob
"""

from keras.models import load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt

model = load_model('first_model.hdf5')
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))

print("\n\n\n\n\n\nSallyBot v0.0.0.0.0.0.0....1\n\n:=================================================\n\n")

while True:
    
    input_str = input("\nAsk Sally a question:\n")
    
    input_seq = tokenizer.texts_to_sequences([input_str])[0]
    
    if len(input_seq) > 10:
        input_seq =  vect[:10]
    else:
        input_seq = input_seq+[0]*(10-len(input_seq))
    
    pred = model.predict(np.array(input_seq).reshape(1,-1))
    
    categories = ['Know about what she has been up to in general.', 
                  'Just say hi!',
                  'Know about Falcons as a species.',
                  'Hear a story of her life']
    
    plt.barh(categories,pred[0])
    
    print(f"\nSally thinks you want to: {categories[np.argmax(pred)]}\n\n")
    
    y=[print(round(x,2)) for x in pred[0]]





