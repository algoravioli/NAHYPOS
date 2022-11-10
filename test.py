#%%
from NAHYPOS import NAHYPOS
import os
from pathlib import Path
import pandas as pd

dataset_list = ["bert", "distilbert", "roberta", "xlnet"]
for i in range(len(dataset_list)):

    FS_DICT = {"POPULATION": 50, "ITERATIONS": 30}
    Searcher = NAHYPOS(FS_DICT=FS_DICT)
    train_data: str = (
        rf"C:\Users\Chris\Desktop\GitFiles\NAHYPOS\Data\{dataset_list[i]}\Train.ftr"
    )

    test_data: str = (
        rf"C:\Users\Chris\Desktop\GitFiles\NAHYPOS\Data\{dataset_list[i]}\Test.ftr"
    )

    (
        TrainData_x,
        TrainData_y,
        TestData_x,
        TestData_y,
        TrainData_np,
        TestData_np,
    ) = Searcher.data_ingest(train_data, test_data, format="FTR")

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
            "Softmax",
        ],
    }
    Searcher.optuna_settings(optuna_dict)
    study, OutputDict = Searcher.run_optuna_study(n_trials=100)
    OptunaResults_df = pd.DataFrame(OutputDict, index=[0])
    OptunaResults_df.to_csv(f"OptunaResults_{dataset_list[i]}.csv", index=False)

    Results = Searcher.run_FS_study(dataset_list[i])

# %%
