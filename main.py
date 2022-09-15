import argparse

import numpy as np

from basis_expansion_checker import basis_expansion_chooser
from utils import all_train_fit, model_predict, mse, read_data, save_predictions


def args_parse():
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--mode', type=str, default=None, choices=["train", "test", None],
                        help='Choose mode for the model train or test.')

    parser.add_argument('--data-analysis', type=bool, default=False,
                        help='Choose if to generate the visualizations from data analysis.')

    args = parser.parse_args()
    return args


def predict():
    Xtrain, ytrain, Xtest = read_data()
    basis = (3, True, True)

    w_ls = all_train_fit(Xtrain, ytrain, basis)
    ytest_preds = model_predict(Xtest, w_ls, basis)

    save_predictions(ytest_preds)


def train():
    np.random.seed(23)
    Xtrain, ytrain, Xtest = read_data()
    basis = basis_expansion_chooser(Xtrain, ytrain)
    w_ls = all_train_fit(Xtrain, ytrain, basis)

    train_loss = mse(ytrain, model_predict(Xtrain, w_ls, basis))
    print(f"Train Loss: {train_loss}")

    ytest_preds = model_predict(Xtest, w_ls, basis)
    save_predictions(ytest_preds)


if __name__ == '__main__':
    args = args_parse()

    print(args)

    if args.mode == "train":
        train()
    elif args.mode == "test":
        predict()

    # if args.data_analysis:
    #     data_analysis()