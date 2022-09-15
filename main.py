from basis_expansion_checker import basis_expansion_chooser
from utils import all_train_fit, model_predict, read_data, normalize_data, remove_feature


if __name__ == '__main__':
    Xtrain, ytrain, Xtest = read_data()

    # Xtrain, ytrain, Xtest = normalize_data(Xtrain, ytrain, Xtest)
    # Xtrain, Xtest = remove_feature(Xtrain, Xtest)

    basis = basis_expansion_chooser(Xtrain, ytrain)
    w_ls = all_train_fit(Xtrain, ytrain, basis)
    ytest_preds = model_predict(Xtest, w_ls, basis)