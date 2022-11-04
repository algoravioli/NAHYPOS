import tensorflow as tf


class LayerAdd:
    def __init__(self, layer_type, model):
        self.layer_type = layer_type
        self.model = model

    def __call__(self, num_units=10, conv_window=3, dropoutrate=0.1):
        if self.layer_type == "Dense":
            self.model.add(tf.keras.layers.Dense(num_units, activation="linear"))
        elif self.layer_type == "CNN":
            self.model.add(tf.keras.layers.Reshape((num_units, 1)))
            self.model.add(
                tf.keras.layers.Conv1D(num_units, conv_window, activation="linear")
            )
        elif self.layer_type == "LSTM":
            self.model.add(tf.keras.layers.Reshape((num_units, 1)))
            self.model.add(tf.keras.layers.LSTM(num_units, activation="linear"))
        elif self.layer_type == "GRU":
            self.model.add(tf.keras.layers.Reshape((num_units, 1)))
            self.model.add(tf.keras.layers.GRU(num_units, activation="linear"))
        elif self.layer_type == "Dropout":
            self.model.add(tf.keras.layers.Dropout(dropoutrate))

        # some activation functions
        elif self.layer_type == "Relu":
            self.model.add(tf.keras.layers.ReLU())
        elif self.layer_type == "LeakyRelu":
            self.model.add(tf.keras.layers.LeakyReLU())
        elif self.layer_type == "PReLU":
            self.model.add(tf.keras.layers.PReLU())
        elif self.layer_type == "ELU":
            self.model.add(tf.keras.layers.ELU())
        elif self.layer_type == "ThresholdedReLU":
            self.model.add(tf.keras.layers.ThresholdedReLU())
        elif self.layer_type == "Softmax":
            self.model.add(tf.keras.layers.Softmax())
        elif self.layer_type == "Sigmoid":
            self.model.add(tf.keras.activations.sigmoid())
        elif self.layer_type == "Tanh":
            self.model.add(tf.keras.activations.tanh())

        # some utility layers
        elif self.layer_type == "BatchNormalization":
            # self.model.add(tf.keras.layers.Reshape((num_units, 1)))
            self.model.add(tf.keras.layers.BatchNormalization())
        elif self.layer_type == "MaxPooling1D":
            # self.model.add(tf.keras.layers.Reshape((num_units, 1)))
            self.model.add(tf.keras.layers.MaxPooling1D())
        elif self.layer_type == "AveragePooling1D":
            self.model.add(tf.keras.layers.AveragePooling1D())
        elif self.layer_type == "GlobalAveragePooling1D":
            self.model.add(tf.keras.layers.GlobalAveragePooling1D())
        elif self.layer_type == "GlobalMaxPooling1D":
            self.model.add(tf.keras.layers.GlobalMaxPooling1D())
        elif self.layer_type == "Flatten":
            self.model.add(tf.keras.layers.Flatten())
        elif self.layer_type == "Reshape":
            self.model.add(tf.keras.layers.Reshape((1, num_units)))
        elif self.layer_type == "UpSampling1D":
            self.model.add(tf.keras.layers.UpSampling1D())
        elif self.layer_type == "ZeroPadding1D":
            self.model.add(tf.keras.layers.ZeroPadding1D())


# class OptimizerAdd:
#     def __init__(self, optimizer_type, model):
#         self.optimizer_type = optimizer_type
#         self.model = model

#     def __call__(self):
#         pass


# EG CODE
# Model = tf.keras.Sequential()
# LayerAdd("Dense", Model)(10)
