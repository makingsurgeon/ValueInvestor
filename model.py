import numpy as np
import pandas as pd 
import os 
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense 

def predict():
    basepath = os.path.dirname(__file__)
    filename = os.path.abspath(os.path.join(basepath, "..","data/raw/2020Q1Q2Q3Q4-2021Q1 - Russia - Sberbank Rossii PAO (SBER).csv"))
    dataset_train = pd.read_csv(filename)
    dataset_train = dataset_train.drop(311)
    dataset_train = dataset_train.iloc[::-1]
    dataset_train = dataset_train.reset_index()
    dataset_train = dataset_train.drop(["index"], axis=1)
    training_set = dataset_train.iloc[:250, 1:2].values
    sc = MinMaxScaler(feature_range=(0,1))
    training_set_scaled = sc.fit_transform(training_set)
    X_train = []
    y_train = []
    for i in range(7, 250):
        X_train.append(training_set_scaled[i-7:i, 0])
        y_train.append(training_set_scaled[i, 0])
    X_train, y_train = np.array(X_train), np.array(y_train)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    model = Sequential()

    model.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50,return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    model.compile(optimizer='adam',loss='mean_squared_error')

    model.fit(X_train,y_train,epochs=100,batch_size=32)

    dataset_total = dataset_train[["Price"]]
    inputs = dataset_total[len(dataset_total) - len(real_stock_price) - 7:].values

    inputs = inputs.reshape(-1,1)
    inputs = sc.transform(inputs)
    X_test = []
    for i in range(7, 68):
        X_test.append(inputs[i-7:i, 0])
    
    X_test = np.array(X_test)
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    predicted_stock_price = model.predict(X_test)
    predicted_stock_price = sc.inverse_transform(predicted_stock_price)

    return predicted_stock_price



