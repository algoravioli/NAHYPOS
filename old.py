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
            num_units = trial.suggest_int("n_units_l{}".format(i), width_min, width_max)
            layer_type = trial.suggest_categorical(
                "layer_type_l{}".format(i), layer_types
            )  # layer types should be in ['Dense', 'Dropout']
            # Add Layers

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
            kwargs["momentum"] = trial.suggest_float("momentum", 1e-5, 1e-1, log=True)
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
