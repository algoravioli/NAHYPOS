#%%
import pandas as pd
import numpy as np
import os

Path = "./FS_results/"
Results = os.listdir(Path)
Results = [i for i in Results if i.endswith(".csv")]

df_all = pd.DataFrame()
# convert results to single dataframe and plot
for i in range(len(Results)):
    df = pd.read_csv(Path + Results[i], index_col=0)
    Accuracy = df["acc"].to_numpy()
    FeatureSet = Results[i].split("_")[2]
    Algorithm = Results[i].split("_")[4]
    Algorithm = Algorithm.split(".")[0]
    NumberOfFeatures = df["number_of_features"].to_numpy()
    df = pd.DataFrame(
        {
            "Accuracy": Accuracy,
            "FeatureSet": FeatureSet,
            "Number Of Features": NumberOfFeatures,
            "Algorithm": Algorithm,
        }
    )
    if i == 0:
        df_all = df
    else:
        df_all = pd.concat([df_all, df])

# plot dataframe
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
ax = sns.catplot(x="FeatureSet", y="Number Of Features", hue="Algorithm", data=df_all)
ax.set(xlabel="Feature Set", ylabel="Accuracy")

plt.show()


#%%
