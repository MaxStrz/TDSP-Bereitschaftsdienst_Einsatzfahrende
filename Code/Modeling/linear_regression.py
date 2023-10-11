import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '../Data_AandU')
import dataPrep as data_prep

#create class for regression
class TrendRegression:
    def __init__(self):
        """Initialize class with a regression model"""
        self.df = pd.read_pickle('df_new_columns.pkl')
        self.df_predict, self.ax, self.reg = data_prep.notruf_reg(self.df)
        self.startdate = np.datetime64('2016-04-01')
        self.df_final_features = self.df_predict.drop(['calls','n_sick', 'demand', 'demand_pred', 'calls_pred'], axis=1)

reg_class = TrendRegression()
reg = reg_class.reg

arr = np.array([[5]])

# Mit Scikit-Learn regression-Modell 'reg', sagen Datum vorher
day_pred = reg.predict(arr)
print(day_pred)