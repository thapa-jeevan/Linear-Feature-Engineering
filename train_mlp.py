import math

import pandas as pd

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error as mse
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import callbacks, losses, optimizers

from mlp import get_model

df_train = pd.read_csv("traindata.txt", sep="   ", names=range(9), engine="python")
df_train = df_train.sample(len(df_train))

ss = StandardScaler()
df_train.iloc[:, :8] = ss.fit_transform(df_train.iloc[:, :8])

sum_cv_test_loss = 0
K = 5
cv = KFold(K)


sum_cv_test_loss = 0

for train_idx, test_idx in cv.split(df_train):
    d_train = df_train.values[train_idx]
    d_test = df_train.values[test_idx]

    X_train, y_train = d_train[:, :-1], d_train[:, -1]
    X_test, y_test = d_test[:, :-1], d_test[:, -1]

    model = get_model()
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