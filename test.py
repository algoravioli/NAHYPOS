#%%
from NAHYPOS import NAHYPOS
import os
from pathlib import Path


Searcher = NAHYPOS()

train_data: str = (
    r"C:\Users\Chris\Desktop\GitFiles\NAHYPOS\Data\roberta\Adresso_Train_TFRoberta.ftr"
)

test_data: str = (
    r"C:\Users\Chris\Desktop\GitFiles\NAHYPOS\Data\roberta\Adresso_Test_TFRoberta.ftr"
)

(
    TrainData_x,
    TrainData_y,
    TestData_x,
    TestData_y,
    TrainData_np,
    TestData_np,
) = Searcher.data_ingest(train_data, test_data, format="FTR")
#%%

optuna_dict = {
    "min_layer_size": 16,
    "max_layer_size": 64,
    "min_layer_number": 1,
    "max_layer_number": 30,
    "allowable_layer_types": [
        "Dense",
    ],
    "output_size": 2,
    "optimizers": ["Adam", "RMSprop", "SGD"],
    "activations": [
        "Relu",
        # "Tanh",
        # "Sigmoid",
        "Softmax",
        # "PReLU",
        # "ELU",
        # "ThresholdedReLU",
    ],
}
Searcher.optuna_settings(optuna_dict)
Searcher.run_optuna_study(n_trials=100)

# %%
