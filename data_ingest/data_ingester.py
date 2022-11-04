import pandas as pd


class Data_Ingester:
    def __init__(self):
        pass

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

        return (
            TrainData_x,
            TrainData_y,
            TestData_x,
            TestData_y,
            TrainData_np,
            TestData_np,
        )

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

        return (
            TrainData_x,
            TrainData_y,
            TestData_x,
            TestData_y,
            TrainData_np,
            TestData_np,
        )
