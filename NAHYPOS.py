#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import optuna
import dehb


class NAHYPOS:
    def __init__(self, EPOCHS):
        super(NAHYPOS, self).__init__()
        self.EPOCHS = EPOCHS

    def DataIngestCSV(self, TrainCSV, TestCSV):
        # import data from csv
        TrainData = pd.read_csv(TrainCSV)
        TestData = pd.read_csv(TestCSV)

        # convert data to np array
        TrainData_np = TrainData.to_numpy()
        TestData_np = TestData.to_numpy()

        # split data into x and y
        TrainData_x = TrainData_np[:, 0:-1]
        TrainData_y = TrainData_np[:, -1]

        TestData_x = TestData_np[:, 0:-1]
        TestData_y = TestData_np[:, -1]

        self.TrainData_x = TrainData_x
        self.TrainData_y = TrainData_y
        self.TestData_x = TestData_x
        self.TestData_y = TestData_y
        self.TrainData_np = TrainData_np
        self.TestData_np = TestData_np

    def DataIngestFTR(self, TrainFTR, TestFTR):
        # import data from ftr
        TrainData = pd.read_feather(TrainFTR)
        TestData = pd.read_feather(TestFTR)

        # convert data to np array
        TrainData_np = TrainData.to_numpy()
        TestData_np = TestData.to_numpy()

        # split data into x and y
        TrainData_x = TrainData_np[:, 0:-1]
        TrainData_y = TrainData_np[:, -1]

        TestData_x = TestData_np[:, 0:-1]
        TestData_y = TestData_np[:, -1]

        self.TrainData_x = TrainData_x
        self.TrainData_y = TrainData_y
        self.TestData_x = TestData_x
        self.TestData_y = TestData_y
        self.TrainData_np = TrainData_np
        self.TestData_np = TestData_np

    def SetupOptuna(
        self,
        trial,
        layer_min,
        layer_max,
        width_min,
        width_max,
        layer_types,
        output_size,
        tf_optimizers,
        tf_loss_fn,
        conv_window=3,
    ):
        # setup optuna
        # define function to create model
        def create_model(
            trial,
            layer_min,
            layer_max,
            width_min,
            width_max,
            layer_types,
            output_size,
            tf_optimizers,
            tf_loss_fn,
            conv_window=3,
        ):
            # define model
            model = tf.keras.models.Sequential()
            # define number of layers
            n_layers = trial.suggest_int("n_layers", layer_min, layer_max)
            # define number of units in each layer
            for i in range(n_layers):
                num_units = trial.suggest_int(
                    "n_units_l{}".format(i), width_min, width_max
                )
                layer_type = trial.suggest_categorical(
                    "layer_type_l{}".format(i), layer_types
                )  # layer types should be in ['Dense', 'Dropout']
                # layers
                if layer_type == "Dense":
                    model.add(tf.keras.layers.Dense(num_units, activation="linear"))
                elif layer_type == "CNN":
                    model.add(
                        tf.keras.layers.Conv1D(
                            num_units, conv_window, activation="linear"
                        )
                    )
                elif layer_type == "LSTM":
                    model.add(tf.keras.layers.LSTM(num_units, activation="linear"))
                elif layer_type == "GRU":
                    model.add(tf.keras.layers.GRU(num_units, activation="linear"))
                elif layer_type == "Dropout":
                    model.add(tf.keras.layers.Dropout(num_units / width_max))

                # some activation functions
                elif layer_type == "Relu":
                    model.add(tf.keras.layers.Activation("relu"))
                elif layer_type == "LeakyRelu":
                    model.add(tf.keras.layers.LeakyReLU())
                elif layer_type == "PReLU":
                    model.add(tf.keras.layers.PReLU())
                elif layer_type == "ELU":
                    model.add(tf.keras.layers.ELU())
                elif layer_type == "ThresholdedReLU":
                    model.add(tf.keras.layers.ThresholdedReLU())
                elif layer_type == "Softmax":
                    model.add(tf.keras.layers.Softmax())
                elif layer_type == "Sigmoid":
                    model.add(tf.keras.layers.Sigmoid())
                elif layer_type == "Tanh":
                    model.add(tf.keras.layers.Tanh())

                # some utility layers
                elif layer_type == "BatchNormalization":
                    model.add(tf.keras.layers.BatchNormalization())
                elif layer_type == "MaxPooling1D":
                    model.add(tf.keras.layers.MaxPooling1D())
                elif layer_type == "AveragePooling1D":
                    model.add(tf.keras.layers.AveragePooling1D())
                elif layer_type == "GlobalAveragePooling1D":
                    model.add(tf.keras.layers.GlobalAveragePooling1D())
                elif layer_type == "GlobalMaxPooling1D":
                    model.add(tf.keras.layers.GlobalMaxPooling1D())
                elif layer_type == "Flatten":
                    model.add(tf.keras.layers.Flatten())
                elif layer_type == "Reshape":
                    model.add(tf.keras.layers.Reshape((1, num_units)))
                elif layer_type == "UpSampling1D":
                    model.add(tf.keras.layers.UpSampling1D())
                elif layer_type == "ZeroPadding1D":
                    model.add(tf.keras.layers.ZeroPadding1D())

            # define output layer
            model.add(tf.keras.layers.Dense(output_size, activation="ReLU"))
            # define optimizer
            kwargs = {}

            optimizer = trial.suggest_categorical(
                "optimizer", tf_optimizers
            )  # optimizers should be in this format: ["Adam", "RMSprop", "SGD"]

            if optimizer == "Adam":
                optimizer = tf.keras.optimizers.Adam(
                    learning_rate=trial.suggest_float("lr", 1e-5, 1e-1)
                )
            elif optimizer == "RMSprop":
                optimizer = tf.keras.optimizers.RMSprop(
                    learning_rate=trial.suggest_float("lr", 1e-5, 1e-1)
                )
                kwargs["decay"] = trial.suggest_float("decay", 0.85, 0.99)
                kwargs["momentum"] = trial.suggest_float(
                    "momentum", 1e-5, 1e-1, log=True
                )
            elif optimizer == "SGD":
                optimizer = tf.keras.optimizers.SGD(
                    learning_rate=trial.suggest_float("lr", 1e-5, 1e-1)
                )
                kwargs["momentum"]
            elif optimizer == "Adagrad":
                optimizer = tf.keras.optimizers.Adagrad(
                    learning_rate=trial.suggest_float("lr", 1e-5, 1e-1)
                )

            optimizer = getattr(tf.keras.optimizers, optimizer)(**kwargs)

            # compile model
            # model.compile(loss=tf_loss_fn, optimizer=optimizer, metrics=["accuracy"])
            return model, optimizer

        # define learn function
        def learn(model, dataset, optimizer, mode="eval"):
            accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)
            for batch, (x, y) in enumerate(dataset):
                with tf.GradientTape() as tape:
                    logits = model(x, training=True)
                    loss_value = tf_loss_fn(y, logits)

                if mode == "eval":
                    accuracy(tf.argmax(logits, axis=1, output_type=tf.int64), y)
                else:
                    grads = tape.gradient(loss_value, model.trainable_weights)
                    optimizer.apply_gradients(zip(grads, model.trainable_weights))
                    if batch % 100 == 0:
                        print("Step #%d\tLoss: %.6f" % (batch, float(loss_value)))

            if mode == "eval":
                return accuracy

        def objective(trial):
            OptunaModel, Optimizer = create_model(
                trial,
                layer_min,
                layer_max,
                width_min,
                width_max,
                layer_types,
                output_size,
                tf_optimizers,
                tf_loss_fn,
                conv_window=3,
            )
            accuracy = tf.metrics.Accuracy("accuracy", dtype=tf.float32)

            # Training and Validating
            for epoch in range(self.EPOCHS):
                learn(OptunaModel, self.TrainData_np, Optimizer, mode="train")

            accuracy = learn(OptunaModel, self.TestData_np, Optimizer, mode="eval")

            return accuracy.result()

        def StartOptunaStudy(self, direction):
            self.study = optuna.create_study(direction=direction)
            self.study.optimize(objective)


# %%

# %%
