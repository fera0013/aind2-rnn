import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Activation
import keras
import re

# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    X = np.asarray([series[i:(i+window_size)] for i in range(series.size-window_size)])
    y = np.asarray(series[window_size:])
    y = np.reshape(y, (len(y),1)) #optional
    return X,y

## TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    model = Sequential()
    model.add(LSTM(5, input_shape = (window_size,1)))
    model.add(Dense(1))
    model.add(Activation('relu'))
    return model

#### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    text = re.sub(r"[^A-Za-z!.:;,?\t]", "", text)
    text.lower()
    return text

#### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = [text[i:window_size] for i in range(0,len(text)-window_size,step_size)]
    outputs = [text[i] for i in range(window_size,len(text)-window_size,step_size+window_size)]
    return inputs,outputs

## TODO build the required RNN model: 
## a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    model = Sequential()
    model.add(LSTM(200, input_shape = (window_size,num_chars)))
    model.add(Dense(num_chars))
    model.add(Activation('softmax'))
    return model
