# import pandas as pd
# import matplotlib.pyplot as plt
import numpy as np
import operator
# import


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, x, y):
        self.x = np.asarray(x)
        self.y = np.asarray(y)

    def _distance(self, v1, v2):
        return np.linalg.norm(v1 - v2)

    def _vote(self, yk):
        vote_dict = {}
        for y in yk:
            if y not in vote_dict.keys():
                vote_dict[y] = 1
            else:
                vote_dict[y] += 1

        sort_vote_dict = sorted(
            vote_dict.items(), key=operator.itemgetter(1), reverse=True
        )
        return sort_vote_dict[0][0]

    def predicate(self, x):
        y_pred = []
        for i in range(len(x)):
            all_dis = [self._distance(x, self.x[j]) for j in range(len(self.x))]
            top_k = np.argsort(all_dis)[: self.k]
            y_pred.append(self._vote(self.y[top_k]))
        return np.asarray(y_pred)

    def score(self, y_preds, y_true):
        if y_preds is None or y_true is None:
            y_preds = self.predicate(self.x)
            y_true = self.y
        y_preds = np.asarray(y_preds)
        y_true = np.asarray(y_true)
        return float(np.mean(y_preds == y_true))
