# Import the necessary packages
import tensorflow as tf
import numpy as np
import re

# Load the bert tokenizer
from transformers import (BertTokenizer, TFBertForSequenceClassification) 
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True, model_max_length=26, truncation=True, return_overflowing_tokens=True, padding_side='left')

model = tf.keras.models.load_model('models/bert') # model location here

def word_space(word):
	# Remove non alphabet characters
	word = re.sub("[^A-Za-z]", "", word.lower())

	# Add spaces between the characters of the word
	return (" ".join(word))

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

def predict_bert(text):
	# Input your word here
	word = word_space(text)
	tokenized = [bert_tokenizer(word,  padding='max_length', truncation=True)['input_ids']]
	prediction = model.predict(tokenized)
	predictions = [np.argmax(x) for x in prediction]
	word_pred = [letter_map[x] for x in predictions]
	return "".join(word_pred).replace(" ", "")