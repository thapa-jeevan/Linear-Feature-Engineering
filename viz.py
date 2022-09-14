import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import math
import numpy as np


def each_x_vs_y(x_train, y_train):
    sub_shape = (2, 4)
    fig, ax = plt.subplots(*sub_shape)
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            fig_idx = sub_shape[1] * i + j
            # print(fig_idx)
            ax[i, j].scatter(x_train[:, fig_idx], y_train)
            ax[i, j].set_title(f'Feature {fig_idx + 1}')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig('each_x_vs_y.png')


def pca_x_vs_y(x_train, y_train):
    sub_shape = (2, 4)
    pca = PCA(n_components=8, svd_solver='full')
    pca.fit(x_train)
    print(pca.explained_variance_ratio_)
    new_x = pca.transform(x_train)
    print(new_x.shape)

    fig, ax = plt.subplots(*sub_shape)
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            fig_idx = sub_shape[1] * i + j
            # print(fig_idx)
            ax[i, j].scatter(new_x[:, fig_idx], y_train)
            ax[i, j].set_title(f'Fea: {fig_idx + 1} \n Var: {int(pca.explained_variance_ratio_[fig_idx] * 100)}')

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)

    plt.savefig('pca_x_vs_y.png')


def get_cov(x_train, y_train):
    print(np.cov(x_train[:, 0], y_train.reshape(-1, 1)))


if __name__ == '__main__':
    df_train = pd.read_csv("traindata.txt", sep="   ", names=range(9), engine="python")
    df_train = df_train.sample(len(df_train))

    x_train = df_train.iloc[:, :-1].values
    y_train = df_train.iloc[:, -1].values.reshape(-1, 1)

    each_x_vs_y(x_train, y_train)
    pca_x_vs_y(x_train, y_train)
    # get_cov(x_train, y_train)





