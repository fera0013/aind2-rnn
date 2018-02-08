import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
import keras


# TODO: fill out the function below that transforms the input series 
# and window-size into a set of input/output pairs for use with our RNN model
def window_transform_series(series, window_size):
    # containers for input/output pairs
    X = [series[i:(i+window_size)] for i in range(series.size)]
    y = series[window_size:]
    # reshape each 
    X = np.asarray(X)
    X.shape = (np.shape(X)[0:2])
    y = np.asarray(y)
    y.shape = (len(y),1)

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


if __name__ == "__main__":
    odd_nums = np.array([1,3,5,7,9,11,13])
    X,y=window_transform_series(odd_nums, 2)
    print '--- the input X will look like ----'
    print X

    print '--- the associated output y will look like ----'
    print y
    print '_'