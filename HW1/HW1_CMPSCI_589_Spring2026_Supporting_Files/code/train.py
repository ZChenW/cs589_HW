from knn import KNN
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


ks = [i for i in range(52) if i % 2 == 1]  # 1~51 的奇数


def read_csv_file(csv_path):
    df = pd.read_csv(csv_path, header=None)
    if df.shape[1] != 31:
        raise ValueError("Incorrect csv formate")
    X = df.iloc[:, :-1].to_numpy(dtype=np.float32)  # 569*30
    y = df.iloc[:, -1].to_numpy(dtype=np.float32)  # 569*1

    assert X.shape[1] == 30  # column
    assert y.shape[0] == X.shape[0]  # Row

    return X, y


def prepare(csv_path, k, seed, normalize=True):
    X, y = read_csv_file(csv_path=csv_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, stratify=y, random_state=seed
    )
    if normalize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    module = KNN(k=k)
    module.fit(X_train, y_train)

    y_test_p = module.predicate(X_test)
    y_train_p = module.train_predicate(X_train)

    test_acc = module.score(y_test_p, y_test)
    train_acc = module.score(y_train_p, y_train)

    return test_acc, train_acc


def train(csv_path):
    base_seed = 0
    train_acc_mean = []
    train_acc_std = []
    test_acc_mean = []
    test_acc_std = []
    for i in ks:
        train_acc_list = []
        test_acc_list = []
        for r in range(20):
            seed = base_seed + 100 * r
            test_acc, train_acc = prepare(csv_path, i, seed)
            train_acc_list.append(train_acc)
            test_acc_list.append(test_acc)
        train_acc_mean.append(np.mean(train_acc_list))
        test_acc_mean.append(np.mean(test_acc_list))
        train_acc_std.append(np.std(train_acc_list, ddof=1))
        test_acc_std.append(np.std(test_acc_list, ddof=1))

    return train_acc_mean, test_acc_mean, train_acc_std, test_acc_std


# print(df.shape)
# print(df.head())
