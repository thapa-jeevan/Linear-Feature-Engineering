import pandas as pd

import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits import mplot3d
import numpy as np
from sklearn.neighbors import KernelDensity
import math

viz_params = {'left': 0.1,
              'bottom': 0.1,
              'right': 0.9,
              'top': 0.9,
              'wspace': 0.4,
              'hspace': 0.7}

def each_x_vs_y(x_train, y_train):
    sub_shape = (2, 4)
    fig, ax = plt.subplots(*sub_shape, figsize=(10, 5))
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            fig_idx = sub_shape[1] * i + j
            # print(fig_idx)
            ax[i, j].scatter(x_train[:, fig_idx], y_train, facecolors='none', edgecolors='#CD5C5C')
            # ax[i, j].set_title(f'Feature {fig_idx + 1}')
            ax[i, j].set_xlabel(f'X{fig_idx + 1}')
            ax[i, j].set_ylabel('Y')

    plt.subplots_adjust(**viz_params)

    plt.savefig('data_viz/each_x_vs_y.png')


def pca_x_vs_y(x_train, y_train):
    sub_shape = (2, 4)
    pca = PCA(n_components=8, svd_solver='full')
    pca.fit(x_train)
    # print(pca.explained_variance_ratio_)
    new_x = pca.transform(x_train)
    # print(new_x.shape)

    fig, ax = plt.subplots(*sub_shape, figsize=(10, 5))
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            fig_idx = sub_shape[1] * i + j
            # print(fig_idx)
            ax[i, j].scatter(new_x[:, fig_idx], y_train, facecolors='none', edgecolors='#CD5C5C')
            # ax[i, j].set_title(f'Component: {fig_idx + 1} \n Var: {round(pca.explained_variance_ratio_[fig_idx] * 100, 2)} %')
            ax[i, j].set_title(f'Var: {round(pca.explained_variance_ratio_[fig_idx] * 100, 2)} %')
            ax[i, j].set_xlabel(f'C{fig_idx + 1}')
            ax[i, j].set_ylabel('Y')
    plt.subplots_adjust(**viz_params)

    plt.savefig('data_viz/pca_x_vs_y.png')


def get_cov(x_train, y_train):
    print(np.cov(x_train[:, 0], y_train.reshape(-1, 1)))


def pca_3d(x_train, y_train):
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(x_train)
    # print(pca.explained_variance_ratio_)
    new_x = pca.transform(x_train)
    # print(new_x.shape)

    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.scatter3D(new_x[:, 0], new_x[:, 1], y_train,
                 c=y_train, cmap='Greens');
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
    plt.savefig('data_viz/pca_3d.png')


def data_distribution(x_train, y_train):
    sub_shape = (3, 3)

    fig, ax = plt.subplots(*sub_shape)
    for i in range(sub_shape[0]):
        for j in range(sub_shape[1]):
            fig_idx = sub_shape[1] * i + j
            X = y_train if (i == 2 and j == 2) else x_train[:, fig_idx]
            title = 'Y' if (i == 2 and j == 2) else f'X{fig_idx + 1}'
            bins = np.linspace(min(X), max(X), 40)

            ax[i, j].hist(list(X.flatten()), bins=bins.flatten(), fc="#AAAAFF",
                          density=True)
            # ax[i, j].set_title(title)
            ax[i, j].set_xlabel(title)

    plt.subplots_adjust(**viz_params)

    # print(y_train.tolist)
    # print(bins)
    # kde = KernelDensity(kernel='gaussian', bandwidth=0.2).fit(X)
    # log_dens = kde.score_samples(X)
    # ax.fill(y_train, np.exp(log_dens), fc="#AAAAFF")

    plt.savefig('data_viz/data_dist.png')


def pca_test_data(x_train, x_test, y_train):
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(x_train)
    new_x_train = pca.transform(x_train)
    new_x_test = pca.transform(x_test)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(new_x_train[:, 0], new_x_train[:, 1], facecolors='none', edgecolors='green', label='Train',)
    # ax.scatter(new_x_train[:, 0], new_x_train[:, 1], c=range(len(y_train)), ec='k', label='Train')
    ax.scatter(new_x_test[:, 0], new_x_test[:, 1],
               facecolors='red',
               # edgecolors='red',
               label='Test',
               marker="x")
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    plt.savefig('data_viz/pca_test_data.png')



def pca_color_map(x_train, x_test, y_train):
    pca = PCA(n_components=2, svd_solver='full')
    pca.fit(x_train)
    new_x_train = pca.transform(x_train)
    new_x_test = pca.transform(x_test)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(new_x_train[:, 0], new_x_train[:, 1], c=range(len(y_train)),
               ec='k',
               cmap='Greens',
               label='Train')
    # ax.scatter(new_x_test[:, 0], new_x_test[:, 1], c=range(len(new_x_test[:, 1])),
    #            ec='k',
    #            cmap='Reds',
    #            label='Test')
    ax.set_title('After Shuffle')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.legend()
    plt.savefig('data_viz/pca_color_map_after.png')


def data_analysis():
    df_train = pd.read_csv("data/traindata.txt", sep="   ", names=range(9), engine="python")
    df_train = df_train.sample(len(df_train))

    x = df_train.iloc[:, :-1].values
    y = df_train.iloc[:, -1].values.reshape(-1, 1)

    X_test = pd.read_csv("data/testinputs.txt", sep="   ", names=range(8), engine="python").values

    each_x_vs_y(x, y)
    pca_x_vs_y(x, y)
    pca_3d(x, y)
    # get_cov(x, y_train)
    data_distribution(x, y)
    pca_test_data(x, X_test, y)
    pca_color_map(x, X_test, y)