#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
import dehb
import optuna_misc.OptunaLayers as OptunaLayers
import data_ingest.data_ingester as data_ingester


class NAHYPOS:
    def __init__(self, EPOCHS=100):
        self.EPOCHS = EPOCHS

    def optuna_settings(self, dict):
        self.optuna_min_layer_size = dict["min_layer_size"]
        self.optuna_max_layer_size = dict["max_layer_size"]
        self.optuna_min_layer_number = dict["min_layer_number"]
        self.optuna_max_layer_numer = dict["optuna_max_layer_number"]   
        self.optuna_allowable_layer_types = dict["allowable_layer_types"]
        self.optuna_output_size = dict["output_size"]
        self.optuna_allowable_optimizers = dict["optimizers"]

    def create_model(self,trial):
        model = tf.keras.Sequential()
        n_layers = trial.suggest_int("n_layers", self.optuna_min_layer_number, self.optuna_max_layer_numer)
        for i in range(n_layers):
            num_units = trial.suggest_int("n_units_l{}".format(i), self.optuna_min_layer_size, self.optuna_max_layer_size)
            layer_type = trial.suggest_categorical("layer_type_l{}".format(i), self.optuna_allowable_layer_types)
            OptunaLayers.LayerAdd(layer_type, model)(num_units)
        
        model.add(tf.keras.layers.Dense(self.optuna_output_size, activation="ReLU"))
    
# %%

Searcher = NAHYPOS()
# Searcher.optuna_settings(optuna_dict)