# Load the necessary packages
import tensorflow as tf
import numpy as np
import re
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer

# Load the lstm model
model = tf.keras.models.load_model('models/lstm') # model location here

# Initialize the characters for tokenization reference
texts = [' ', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']

# Creating a tokenizer
tokenizer = Tokenizer(lower=True)

# Building word indices
tokenizer.fit_on_texts(texts)

# Create the preprocessor
def split_word(word):
	# Remove non alphabet characters
	word = re.sub("[^A-Za-z]", "", word.lower())
	# Split the words into characters
	return list(word) 

# Declare the letter map dictionary
letter_map = {' ':0, 'a':1, 'b':2, 'c':3, 'd':4, 'e':5, 
                'f':6, 'g':7, 'h':8, 'i':9, 'j':10,
                'k':11, 'l':12, 'm':13, 'n': 14, 'o':15,
                'p':16, 'q':17, 'r':18, 's':19, 't':20,
                'u':21, 'v':22, 'w':23, 'x':24, 'y':25, 'z':26,
                0:' ', 1:'a', 2:'b', 3:'c', 4:'d', 5:'e',
                6:'f', 7:'g', 8:'h', 9:'i', 10:'j',
                11:'k', 12:'l', 13:'m', 14:'n', 15:'o',
                16:'p', 17:'q', 18:'r', 19:'s', 20:'t',
                21:'u', 22:'v', 23:'w', 24:'x', 25:'y', 26:'z'
               }

def predict_lstm(text):
	# Input your word here
	trial = (tokenizer.texts_to_sequences([split_word(text)]))
	trial = pad_sequences(trial, maxlen=24)
	prediction = model.predict(trial.reshape(1, -1))
	predictions = [np.argmax(x) for x in prediction]
	word_pred = [letter_map[x] for x in predictions]
	return "".join(word_pred).replace(" ", "")