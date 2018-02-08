import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    X = np.asarray([series[i:(i+window_size)] for i in range(series.size-window_size)])
    y = np.asarray(series[window_size:])
    y = np.reshape(y, (len(y),1)) #optional
    return X,y

## TODO: build an RNN to perform regression on our time series input/output data
def build_part1_RNN(window_size):
    pass


#### TODO: return the text input with only ascii lowercase and the punctuation given below included.
def cleaned_text(text):
    punctuation = ['!', ',', '.', ':', ';', '?']

    return text

#### TODO: fill out the function below that transforms the input text and window-size into a set of input/output pairs for use with our RNN model
def window_transform_text(text, window_size, step_size):
    # containers for input/output pairs
    inputs = []
    outputs = []

    return inputs,outputs

## TODO build the required RNN model: 
## a single LSTM hidden layer with softmax activation, categorical_crossentropy loss 
def build_part2_RNN(window_size, num_chars):
    pass

