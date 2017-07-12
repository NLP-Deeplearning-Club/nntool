import random
import math
import linecache
import numpy as np

class DataSet:
    X = None
    y = None
    names = None
    size = None

    def __init__(self, names, X, y):
        self.names = names
        self.X = X
        self.y = y
        self.size = len(y)

    def sample(self, k=10):
        if isinstance(k, int) and k > 0:
            l = k
        elif isinstance(k, float) and 0 < k < 1:
            l = math.ceil(self.size * k)
        else:
            raise AttributeError("unsupport arg")

        indexs = random.sample(range(self.size), l)
        X = np.array([self.X[i] for i in indexs])
        y = np.array([self.y[i] for i in indexs])
        return indexs, DataSet(self.names, X, y)

    def randomsplit(self, rate=0.3):
        indexs, sampleset_f = self.sample(rate)
        other_indexs = list(set(range(self.size)) - set(indexs))
        X_s = np.array([self.X[i] for i in other_indexs])
        y_s = np.array([self.y[i] for i in other_indexs])
        sampleset_s = DataSet(self.names, X_s, y_s)
        return sampleset_f, sampleset_s


class DataLoader:
    def __init__(self, namespath, datapath):
        self.namespath = namespath
        self.datapath = datapath
        self._loaded = False

    @property
    def loaded(self):
        return self._loaded

    def load(self):
        names = self._load_names()
        X, y = self._load_datas()
        dataset = DataSet(names, X, y)
        self._loaded = True
        return dataset

    def _load_names(self):
        with open(self.namespath) as f:
            names = [i for i in f][:-1]
        return names

    def _load_datas(self):
        X, y = [], []
        temp = linecache.getlines(self.datapath)
        for line in temp:
            line_word = line.strip().lstrip().split(",")
            y_ = line_word[-1]
            x = [float(i) for i in line_word[:-1]]
            X.append(x)
            y.append(y_)
        X = np.array(X)
        y = np.array(y)
        return X, y
