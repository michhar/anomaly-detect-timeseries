""" 
Based closely upon
https://github.com/aurotripathy/lstm-anomaly-detect
which is inspired by example from
https://github.com/Vict0rSch/deep_learning/tree/master/keras/recurrent

Uses the Keras API within TensorFlow 2
The basic idea is to detect anomalies in synthetic, normalized 
time-series data in an unsupervised manner.
"""
import time
from datetime import datetime as dt
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from numpy import arange, sin, pi, random
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from math import sqrt
from _config import Config

# Global hyper-parameters
config = Config("config.yaml")

sequence_length = 100
random_data_dup = 10  # each sample randomly duplicated between 0 and 9 times, see dropin function
mse_threshold = 0.1 # anomaly MSE threshold

train_start, train_end, test_start, test_end = (0, 700, 500, 1000)

def make_dirs(_id):
    '''Create directories for storing data in repo (using datetime ID) if they don't already exist'''

    # if not config.train or not config.predict:
    #     if not os.path.isdir('data/%s' %config.use_id):
    #         raise ValueError("Run ID %s is not valid. If loading prior models or predictions, must provide valid ID.")

    paths = ['data', 'data/%s' %_id, 'data/%s/models' %_id, 
        'data/%s/smoothed_errors' %_id, 'data/%s/y_hat' %_id]

    for p in paths:
        if not os.path.isdir(p):
            os.mkdir(p)

def dropin(X, y):
    """ The name suggests the inverse of dropout, i.e. adding more samples. See Data Augmentation section at
    http://simaaron.github.io/Estimating-rainfall-from-weather-radar-readings-using-recurrent-neural-networks/
    :param X: Each row is a training sequence
    :param y: Tne target we train and will later predict
    :return: new augmented X, y
    """
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    X_hat = []
    y_hat = []
    for i in range(0, len(X)):
        for j in range(0, np.random.random_integers(0, random_data_dup)):
            X_hat.append(X[i, :])
            y_hat.append(y[i])
    return np.asarray(X_hat), np.asarray(y_hat)


def gen_wave():
    """ Generate a synthetic wave by adding up a few sine waves and some noise
    :return: the final wave
    """
    t = np.arange(0.0, 10.0, 0.01)
    wave1 = sin(2 * 2 * pi * t)
    noise = random.normal(0, 0.1, len(t))
    wave1 = wave1 + noise
    print("wave1", len(wave1))
    wave2 = sin(2 * pi * t)
    print("wave2", len(wave2))
    t_rider = arange(0.0, 0.5, 0.01)
    wave3 = sin(10 * pi * t_rider)
    print("wave3", len(wave3))
    insert = round(0.8 * len(t))
    wave1[insert:insert + 50] = wave1[insert:insert + 50] + wave3
    return wave1 + wave2

def load_data():
    '''Load train and test data from repo. If not in repo need to download from source.
    Args:
        anom (dict): contains anomaly information for a given input stream
    Returns:
        X_train (np array): array of train inputs with dimensions [timesteps, l_s, input dimensions]
        y_train (np array): array of train outputs corresponding to true values following each sequence
        X_test (np array): array of test inputs with dimensions [timesteps, l_s, input dimensions)
        y_test (np array): array of test outputs corresponding to true values following each sequence
    '''
    data = gen_wave()
    train = data[train_start:train_end]
    test = data[test_start:test_end]



    # shape, split data
    X_train, y_train = shape_data(train)
    X_test, y_test = shape_data(test, train=False)

    print("Shape X_train", np.shape(X_train))
    print("Shape X_test", np.shape(X_test))

    # To standardize (StandardScaler) or normalize (MinMaxScaler)
    # on training data (try each - observe loss)
    # scaler = StandardScaler() # This assumes a Gaussian distribution
    # scaler = MinMaxScaler(feature_range=(0, 1)) # Normalize from 0 - 1
    # scaler = scaler.fit_transform(X_train)
    # X_test = scaler.transform(X_test)


    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    return X_train, y_train, X_test, y_test


def shape_data(arr, train=True):
    '''Shape raw input streams for ingestion into LSTM. config.l_s specifies the sequence length of 
    prior timesteps fed into the model at each timestep t. 
    Args:
        arr (np array): array of input streams with dimensions [timesteps, 1, input dimensions]
        train (bool): If shaping training data, this indicates data can be shuffled
    Returns:
        X (np array): array of inputs with dimensions [timesteps, l_s, input dimensions)
        y (np array): array of outputs corresponding to true values following each sequence. 
            shape = [timesteps, n_predictions, 1)
        l_s (int): sequence length to be passed to test shaping (if shaping train) so they are consistent
    '''
    
    # print("LEN ARR: %s" %len(arr))

    data = [] 
    for i in range(len(arr) - config.l_s - config.n_predictions):
        data.append(arr[i:i + config.l_s + config.n_predictions])
    data = np.array(data) 

    # data = data[:, :]

    if train == True:
        np.random.shuffle(data)

    X = data[:,:-config.n_predictions]
    y = data[:,-config.n_predictions:] #telemetry value is at position 0

    return X, y

def build_model(anom, X_train, y_train):
    
    cbs = [tf.keras.callbacks.History(), tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=config.patience, 
        min_delta=config.min_delta, verbose=0)]

    model = tf.keras.Sequential()
    # layers = [1, 50, 100, 1]

    model.add(tf.keras.layers.LSTM(
        config.layers[0],
        input_shape=(None, 1), # could update 1 to n for multivariate
        return_sequences=True))
    model.add(tf.keras.layers.Dropout(config.dropout))

    model.add(tf.keras.layers.LSTM(
        config.layers[1],
        return_sequences=False))
    model.add(tf.keras.layers.Dropout(config.dropout))

    model.add(tf.keras.layers.Dense(
        config.n_predictions))
    model.add(tf.keras.layers.Activation("linear"))

    start = time.time()
    # Loss function and optimizer (SGD-based)
    model.compile(loss=config.loss_metric, optimizer=config.optimizer) # also can try:  optimizer="adam"
    model.fit(X_train, y_train, batch_size=config.lstm_batch_size, epochs=config.epochs, 
        validation_split=config.validation_split, callbacks=cbs, verbose=True)
    model.save(os.path.join("data", anom['run_id'], "models", "model.h5"))

    return model

def predict_in_batches(X_test, y_test, model, anom):
    '''Used trained LSTM model to predict test data arriving in batches (designed to 
    mimic a spacecraft downlinking schedule).
    Args:
        y_test (np array): numpy array of test outputs corresponding to true values to be predicted at end of each sequence
        X_test (np array): numpy array of test inputs with dimensions [timesteps, l_s, input dimensions)
        model (obj): trained Keras model 
        anom (dict): contains all anomaly information for a given input stream
    Returns:
        y_hat (np array): predicted test values for each timestep in y_test  
    '''

    y_hat = np.array([])

    num_batches = int((y_test.shape[0] - config.l_s) / config.batch_size)
    if num_batches < 0:
        raise ValueError("l_s (%s) too large for stream with length %s." %(config.l_s, y_test.shape[0]))

    # simulate data arriving in batches
    for i in range(1, num_batches+2):
        prior_idx = (i-1) * config.batch_size
        idx = i * config.batch_size
        if i == num_batches+1:
            idx = y_test.shape[0] #remaining values won't necessarily equal batch size
        
        X_test_period = X_test[prior_idx:idx]

        y_hat_period = model.predict(X_test_period)

        # map predictions n steps ahead to their corresponding timestep
        # TODO: vectorize
        final_y_hat = []
        for t in range(len(y_hat_period)+config.n_predictions):
            y_hat_t = []
            for j in range(config.n_predictions):
                if t - j >= 0 and t-j < len(y_hat_period):
                    y_hat_t.append(y_hat_period[t-j][j])
            if t < len(y_hat_period):
                if y_hat_t.count(0) == len(y_hat_t):
                    final_y_hat.append(0)
                else:
                    final_y_hat.append(y_hat_t[0]) # first prediction


        y_hat_period = np.array(final_y_hat).reshape(len(final_y_hat),1)
        y_hat = np.append(y_hat, y_hat_period)

    try:
        print('Shape y_test: ', y_test.shape, ' Shape of y_hat: ', y_hat.shape)
        # plt.savefig(os.path.join("data", anom['run_id'], 'Results.png'))
        plt.subplot(311)
        plt.title("Actual Test Signal w/Anomalies")
        # Just look at first value in period
        plt.plot(y_test[:len(y_test), 0], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal")
        plt.plot(y_hat[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Mean Squared Error")
        mse = ((y_test[:, 0] - y_hat) ** 2)
        plt.plot(mse, 'r')
        plt.savefig(os.path.join("data", anom['run_id'], 'Results.png'))
    except Exception as e:
        print("plotting exception")
        print(str(e))

    y_hat = np.reshape(y_hat, (y_hat.size,))

    np.save(os.path.join("data", anom['run_id'], "y_hat", "y_hat.npy"), np.array(y_hat))



    return y_hat

def run_network(model=None, data=None):
    global_start_time = time.time()

    if data is None:
        print('Loading data... ')
        # Train on first 700 samples and test on next 300 samples (has anomaly)
        # X_train, y_train, X_test, y_test = get_split_prep_data(0, 700, 500, 1000)
        X_train, y_train, X_test, y_test = load_data()
    else:
        X_train, y_train, X_test, y_test = data

    print('\nData Loaded. Compiling...\n')

    if model is None:
        model = build_model(X_train, y_train)

    cbs = [tf.keras.callbacks.History(), tf.keras.EarlyStopping(monitor='val_loss', patience=config.patience, 
        min_delta=config.min_delta, verbose=0)]

    try:
        print("Training...")
        model.fit(X_train, y_train, batch_size=config.lstm_batch_size, epochs=config.epochs, 
            validation_split=config.validation_split, callbacks=cbs, verbose=True)
        print("Predicting...")
        predicted = model.predict(X_test)
        print("Reshaping predicted")
        predicted = np.reshape(predicted, (predicted.size,))
    except KeyboardInterrupt:
        print("prediction exception")
        print('Training duration (s) : ', time.time() - global_start_time)
        return model, y_test, 0

    try:
        plt.savefig('result.png')
        plt.subplot(311)
        plt.title("Actual Test Signal w/Anomalies")
        plt.plot(y_test[:len(y_test)], 'b')
        plt.subplot(312)
        plt.title("Predicted Signal")
        plt.plot(predicted[:len(y_test)], 'g')
        plt.subplot(313)
        plt.title("Mean Squared Error")
        mse = ((y_test - predicted) ** 2)
        plt.plot(mse, 'r')
        plt.savefig('result.png', bbox_inches='tight')
    except Exception as e:
        print("plotting exception")
        print(str(e))

    print('Training duration (s) : ', time.time() - global_start_time)

    print("Anomalies above MSE threshold:  ", np.where(mse > mse_threshold))

    return model, y_test, predicted

if __name__ == '__main__':
    anom = {} # could be data here with DictReader or data_gen
    _id = dt.now().strftime("%Y-%m-%d_%H.%M.%S")
    make_dirs(_id)
    anom['run_id'] = _id
    X_train, y_train, X_test, y_test = load_data()
    model = build_model(anom, X_train, y_train)
    predict_in_batches(X_test, y_test, model, anom)