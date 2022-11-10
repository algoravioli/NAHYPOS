from external_libraries.FS.ba import jfs as bajfs
from external_libraries.FS.cs import jfs as csjfs
from external_libraries.FS.de import jfs as dejfs
from external_libraries.FS.fa import jfs as fajfs
from external_libraries.FS.fpa import jfs as fpajfs
from external_libraries.FS.ga import jfs as gajfs
from external_libraries.FS.gwo import jfs as gwojfs
from external_libraries.FS.hho import jfs as hhojfs
from external_libraries.FS.ja import jfs as jajfs
from external_libraries.FS.pso import jfs as psojfs
from external_libraries.FS.sca import jfs as scajfs
from external_libraries.FS.ssa import jfs as ssajfs
from external_libraries.FS.woa import jfs as woajfs

import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt

import time


class useFS:
    def __init__(self, POPULATIONS, ITERATIONS):
        # opts default
        self.k = 10  # k value in KNN

        self.N = POPULATIONS  # population size
        self.T = ITERATIONS  # max number of iterations

    def run_ba(self):
        fmax = 2  # maximum frequency
        fmin = 0  # minimum frequency
        alpha = 0.9  # constant
        gamma = 0.9  # constant
        A = 2  # maximum loudness
        r = 1  # maximum pulse rate
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "fmax": fmax,
            "fmin": fmin,
            "alpha": alpha,
            "gamma": gamma,
            "A": A,
            "r": r,
        }

        fmdl = bajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_cs(self):
        Pa = 0.25  # discovery rate
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T, "Pa": Pa}
        fmdl = csjfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_de(self):
        CR = 0.9  # crossover rate
        F = 0.5  # constant factor
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "CR": CR,
            "F": F,
        }
        fmdl = dejfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_fa(self):
        alpha = 1  # constant
        beta0 = 1  # light amplitude
        gamma = 1  # absorbtion coefficient
        theta = 0.97  # control alpha
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "alpha": alpha,
            "beta0": beta0,
            "gamma": gamma,
            "theta": theta,
        }
        fmdl = fajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_fpa(self):
        P = 0.8  # switch probability
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T, "P": P}
        fmdl = fpajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_ga(self):
        CR = 0.8  # crossover rate
        MR = 0.01  # mutation rate
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "CR": CR,
            "MR": MR,
        }
        fmdl = gajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_gwo(self):
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T}
        fmdl = gwojfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_hho(self):
        beta = 1.5  # levy component
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T, "beta": beta}
        fmdl = hhojfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_ja(self):
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T}
        fmdl = jajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_pso(self):
        c1 = 2  # cognitive factor
        c2 = 2  # social factor
        w = 0.9  # inertia weight
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "c1": c1,
            "c2": c2,
            "w": w,
        }
        fmdl = psojfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_sca(self):
        alpha = 2
        opts = {
            "k": self.k,
            "fold": self.fold,
            "N": self.N,
            "T": self.T,
            "alpha": alpha,
        }
        fmdl = scajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_ssa(self):
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T}
        fmdl = ssajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def run_woa(self):
        b = 1  # constant
        opts = {"k": self.k, "fold": self.fold, "N": self.N, "T": self.T, "b": b}
        fmdl = woajfs(self.train_data_x, self.train_data_y, opts)
        sf = fmdl["sf"]

        return fmdl, opts, sf

    def declare_dataset(self, train_data, test_data):  # train and test data in np
        self.train_data_x = train_data[:, :-1]
        self.train_data_y = train_data[:, -1]
        self.test_data_x = test_data[:, :-1]
        self.test_data_y = test_data[:, -1]
        self.fold = {
            "xt": self.train_data_x,
            "yt": self.train_data_y,
            "xv": self.test_data_x,
            "yv": self.test_data_y,
        }

    def optimize(self, fmdl, opts, sf, name):
        AlgorithmName = name.split("_")[1]
        AlgorithmName = AlgorithmName.upper()
        num_train = np.size(self.train_data_x, 0)
        num_test = np.size(self.test_data_x, 0)
        x_train = self.train_data_x[:, sf]
        y_train = self.train_data_y.reshape(num_train)
        x_test = self.test_data_x[:, sf]
        y_test = self.test_data_y.reshape(num_test)

        model = KNeighborsClassifier(n_neighbors=self.k)
        model.fit(x_train, y_train)

        y_pred = model.predict(x_test)
        acc = np.sum(y_pred == y_test) / num_test
        print(f"Accuracy of {AlgorithmName}: ", acc * 100)

        number_of_features = fmdl["nf"]
        print(f"Feature Size of {AlgorithmName}: ", number_of_features)

        curve = fmdl["c"]
        curve = curve.reshape(np.size(curve, 1))
        x = np.arange(0, opts["T"], 1.0) + 1.0

        fig, ax = plt.subplots()
        ax.plot(x, curve, "o-")
        ax.set_xlabel("Number of Iterations")
        ax.set_ylabel("Fitness")
        ax.set_title(f"{AlgorithmName}")
        ax.grid()
        plt.show()
        epoch_time = int(time.time())
        fig.savefig(f"./ImageOutput/{AlgorithmName}_{epoch_time}.png")

        return acc, number_of_features
