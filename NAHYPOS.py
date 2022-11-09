#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
import dehb
import optuna_misc.OptunaLayers as OptunaLayers
from data_ingest.Data_Ingester import Data_Ingester

from tqdm import tqdm

# from external_libraries.FS


class NAHYPOS:
    def __init__(self, EPOCHS=100, BATCH_SIZE=100):
        self.EPOCHS = EPOCHS
        self.BATCH_SIZE = BATCH_SIZE

    def data_ingest(self, train_data, test_data, format="CSV"):
        data_ingester = Data_Ingester()
        if format == "CSV":
            (
                TrainData_x,
                TrainData_y,
                TestData_x,
                TestData_y,
                TrainData_np,
                TestData_np,
            ) = data_ingester.DataIngestCSV(train_data, test_data)
        elif format == "FTR":
            (
                TrainData_x,
                TrainData_y,
                TestData_x,
                TestData_y,
                TrainData_np,
                TestData_np,
            ) = data_ingester.DataIngestFTR(train_data, test_data)
        else:
            raise ValueError("Format not supported")

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (TrainData_x, TrainData_y.astype(int))
        )
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(self.BATCH_SIZE)

        test_dataset = tf.data.Dataset.from_tensor_slices(
            (TestData_x, TestData_y.astype(int))
        )
        test_dataset = test_dataset.shuffle(buffer_size=1024).batch(self.BATCH_SIZE)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.input_size = TrainData_x.shape[1]

        return (
            TrainData_x,
            TrainData_y,
            TestData_x,
            TestData_y,
            TrainData_np,
            TestData_np,
        )

    def optuna_settings(self, dict):
        self.optuna_min_layer_size = dict["min_layer_size"]
        self.optuna_max_layer_size = dict["max_layer_size"]
        self.optuna_min_layer_number = dict["min_layer_number"]
        self.optuna_max_layer_numer = dict["max_layer_number"]
        self.optuna_allowable_layer_types = dict["allowable_layer_types"]
        self.optuna_output_size = dict["output_size"]
        self.optuna_allowable_optimizers = dict["optimizers"]
        self.optuna_allowable_activations = dict["activations"]

    def create_model(self, trial):
        model = tf.keras.Sequential()
        n_layers = trial.suggest_int(
            "n_layers", self.optuna_min_layer_number, self.optuna_max_layer_numer
        )
        for i in range(n_layers):
            num_units = trial.suggest_int(
                "n_units_l{}".format(i),
                self.optuna_min_layer_size,
                self.optuna_max_layer_size,
            )

            if i == 0:

                OptunaLayers.LayerAdd("Dense", model)(num_units)
                # model.add(tf.keras.layers.Reshape((num_units, 1)))
                activation_type = trial.suggest_categorical(
                    "activation_l{}".format(i), self.optuna_allowable_activations
                )
                OptunaLayers.LayerAdd(activation_type, model)(num_units)
            else:

                layer_type = trial.suggest_categorical(
                    "layer_type_l{}".format(i), self.optuna_allowable_layer_types
                )
                OptunaLayers.LayerAdd(layer_type, model)(num_units)

                activation_type = trial.suggest_categorical(
                    "activation_l{}".format(i), self.optuna_allowable_activations
                )
                OptunaLayers.LayerAdd(activation_type, model)(num_units)
        # model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(self.optuna_output_size, activation="softmax"))
        return model

    def create_optimizer(self, trial):
        learning_rate = trial.suggest_float("lr", 1e-5, 1e-1)
        optimizer = trial.suggest_categorical(
            "optimizer", self.optuna_allowable_optimizers
        )  # optimizers should be in this format: ["Adam", "RMSprop", "SGD"]
        if optimizer == "RMSprop":
            kwargs = {}
            kwargs["decay"] = trial.suggest_float("decay", 0.85, 0.99)
            kwargs["momentum"] = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        elif optimizer == "SGD":
            kwargs = {}
            kwargs["momentum"] = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
        else:
            kwargs = {}

        optimizer = getattr(tf.keras.optimizers, optimizer)(**kwargs)
        return optimizer

    def learn(self, model, optimizer, dataset, mode="eval"):
        accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

        for batch, (x, y) in enumerate(dataset):
            print(batch, end="\r")
            plt.close()
            with tf.GradientTape() as tape:
                logits = model(x, training=(mode == "train"))
                # tf.print("output: ", logits)
                loss_value = tf.reduce_mean(
                    tf.nn.sparse_softmax_cross_entropy_with_logits(
                        logits=logits, labels=y
                    )
                )

                if mode == "eval":
                    tf.print("output: ", logits)
                    tf.print("target: ", y)
                    plt.plot(logits)
                    plt.plot(y)
                    plt.show()
                    accuracy(
                        tf.argmax(logits, axis=1, output_type=tf.int64),
                        tf.cast(y, tf.int64),
                    )
                    # tf.print("output: ", logits)
                    # tf.print("target: ", y)
                    # plt.plot(logits)
                    # plt.plot(y)
                    # plt.show()
                else:
                    grads = tape.gradient(loss_value, model.variables)
                    optimizer.apply_gradients(zip(grads, model.variables))

        if mode == "eval":
            return accuracy

    def objective(self, trial):
        model = self.create_model(trial)
        optimizer = self.create_optimizer(trial)

        for _ in tqdm(range(self.EPOCHS)):
            print(self.EPOCHS, end="\r")
            self.learn(model, optimizer, self.train_dataset, mode="train")

        accuracy = self.learn(model, optimizer, self.test_dataset, mode="eval")
        assert accuracy is not None
        return accuracy.result()

    def run_optuna_study(self, n_trials=100):
        study = optuna.create_study(direction="maximize")
        study.optimize(self.objective, n_trials=n_trials)

        print("Number of finished trials: ", len(study.trials))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        return study


# %%

# Searcher = NAHYPOS()
# train_data = "./Data/roberta/Adresso_Test_TFRoberta.ftr"
# test_data = "./Data/roberta/Addresso_Test_TFRoberta.ftr"
# Searcher.data_ingest(train_data, test_data, format="FTR")

# optuna_dict = {
#     "min_layer_size": 16,
#     "max_layer_size": 256,
#     "min_layer_number": 1,
#     "max_layer_number": 3,
#     "allowable_layer_types": ["Dense", "Conv1D", "Conv2D"],
#     "output_size": 2,
#     "optimizers": ["Adam", "RMSprop", "SGD"],
# }
# Searcher.optuna_settings(optuna_dict)
# Searcher.run_optuna_study(n_trials=100)

#%%
