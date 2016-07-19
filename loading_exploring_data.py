import pandas as pd
import matplotlib.pyplot as plt
import numpy as numpy
from keras.models import Sequential
from keras.layers import Dense

def load_prep_data(filepath):
    #read Data Frame in and convert to numpy array
    df = pd.read_csv(filepath, usecols=[1], delimiter=';')
    df.dropna(inplace = True)
    data = df.values
    data = data.astype('float32')
    return data

def plot_data(df):
    #Inital EDA -- not very detailed, I know
    plt.plot(df)
    plt.show()
    pass

def train_test_split(df):
    """Given that this is time series cross-validation doesn't work well.
    We need to preserve order of the data given that it's time series
    Splitting the data so that 2/3 is available to train and 1/3 is availble to test is a good approach"""
    training_size = int(len(df) * .67)
    test_size = int(len(df) - training_size)
    train, test = df[0:training_size], df[training_size:len(df)]
    return train, test

def feature_engineer_dataset(data, look_back = 1):
    """Create a dataset so that each observation has s the previous X obervations seat count"""
    dataX = []
    dataY = []
    est_range = len(data)-look_back-1
    for i in range(est_range):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i + look_back, 0])

    return numpy.array(dataX), numpy.array(dataY)

def train_the_nn(features, label, look_back = 1):
    model = Sequential()
    model.add(Dense(8, input_dim=look_back, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer = 'adam')
    model.fit(features, label, nb_epoch=200, batch_size=2, verbose=2)
    return model

def assess_nn(nn, train_feat, train_label, test_feat, test_label):
    #this model hasn't been optimized yet, but it's a starting point
    trainScore = nn.evaluate(train_feat, train_label, verbose=0)
    print 'Train Score', trainScore
    testScore = nn.evaluate(test_feat, test_label, verbose=0)
    print 'Test Score', testScore

if __name__ == '__main__':
    data_prep = load_prep_data('/Users/marcversage/Desktop/Apps/Predicting-Airline-Passengers-Time-Series-NN/data/international-airline-passengers.csv')
    plot_data(data_prep)
    train, test = train_test_split(data_prep)
    look_back = 1
    trainX, trainY = feature_engineer_dataset(train, look_back)
    testX, testY = feature_engineer_dataset(test, look_back)
    nn_model = train_the_nn(trainX, trainY)
    assess_nn(nn_model, trainX, trainY, testX, testY)
