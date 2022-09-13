from basis_expansion_checker import basis_expansion_chooser
from utils import all_train_fit, model_predict, read_data


if __name__ == '__main__':
    Xtrain, ytrain, Xtest = read_data()
    basis = basis_expansion_chooser(Xtrain, ytrain)
    w_ls = all_train_fit(Xtrain, ytrain, basis)
    ytest_preds = model_predict(Xtest, w_ls, basis)