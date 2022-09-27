import math
import tensorflow as tf
import itertools

import numpy as np
import pandas as pd

from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, models, layers, losses, optimizers


df_train = pd.read_csv("traindata.txt", sep="   ", names=range(9), engine="python")
df_train = df_train.sample(len(df_train))

ss = StandardScaler()
df_train.iloc[:, :8] = ss.fit_transform(df_train.iloc[:, :8])

sum_cv_test_loss = 0
K = 5
cv = KFold(K)


class ResBlock(layers.Layer):
    def __init__(self, num_hidden):
        super(ResBlock, self).__init__()
        self.l1 = layers.Dense(num_hidden, activation='relu')
        self.bn1 = layers.BatchNormalization()
        self.l2 = layers.Dense(num_hidden)
        self.relu = layers.ReLU()
        self.add = layers.Add()
        self.bn2 = layers.BatchNormalization()

    def call(self, x):
        fx = self.l1(x)
        fx = self.bn1(fx)
        fx = self.l2(fx)
        out = self.add([x, fx])
        out = self.relu(out)
        out = self.bn2(out)
        return out


sum_cv_test_loss = 0

for train_idx, test_idx in cv.split(df_train):
    d_train = df_train.values[train_idx]
    d_test = df_train.values[test_idx]

    X_train, y_train = d_train[:, :-1], d_train[:, -1]
    X_test, y_test = d_test[:, :-1], d_test[:, -1]

    model = models.Sequential([
        layers.Dense(256, activation="relu"),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        ResBlock(256),
        # ResBlock(256),
        # ResBlock(256),
        # ResBlock(256),
        layers.Dense(1),
    ])

    initial_learning_rate = 0.003


    def lr_exp_decay(epoch, lr):
        k = 0.005
        return initial_learning_rate * math.exp(-k * epoch)


    model.compile(
        optimizer=optimizers.Adam(learning_rate=initial_learning_rate),
        loss=losses.MeanSquaredError()
    )

    model.fit(
        X_train,
        y_train,
        batch_size=256,
        epochs=500,
        validation_data=(X_test, y_test),
        callbacks=[
            callbacks.LearningRateScheduler(lr_exp_decay, verbose=0),
            callbacks.EarlyStopping(patience=20, verbose=0)
        ],
        verbose=1,
    )

    print(y_train.shape, model.predict(X_train).shape)
    mse_train = mse(y_train, model.predict(X_train))
    mse_test = mse(y_test, model.predict(X_test))
    print(f"Train MSE = {mse_train}, Test MSE = {mse_test}")
    sum_cv_test_loss += mse_test

print(sum_cv_test_loss / K)