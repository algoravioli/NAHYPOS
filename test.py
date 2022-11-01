#%%
import NAHYPOS

Search = NAHYPOS.NAHYPOS(100)

Search.DataIngestFTR("Adresso_Train_TFRoberta.ftr", "Adresso_Test_TFRoberta.ftr")

print(Search.TrainData_np)

# %%
