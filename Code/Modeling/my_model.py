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
        self.reg_steigerung = self.reg.coef_
        self.reg_intercept = self.reg.intercept_
        self.last_train_date = np.datetime64(self.df['date'].max())


    def predict(self):
        """Vorhersage der Anzahl der Notrufe"""
        future = pd.date_range(start=self.last_train_date, periods=366, freq='D', inclusive='right')
        future_as_ndays = np.array((future - self.startdate).days + 1)
        # reshape to 2D array for sklearn
        future_as_ndays = future_as_ndays.reshape(-1,1)
        y = self.reg.predict(future_as_ndays)
        return y, future

Reg_Class = TrendRegression()
y, future = Reg_Class.predict()
print(y)

class RandomForest:
    def __init__(self, future):
        """Klasse für Random Forest"""
        self.df = pd.read_pickle('df_new_columns.pkl')
        self.df_final_features = self.df.drop(['calls', 'n_sick', 'demand', 'demand_pred', 'calls_pred'], axis=1)
        self.full_pred_df, self.feat_gini_importance, self.results_df, self.adabr = data_prep.my_model_options(self.df_final_features)
        self.df2 = pd.DataFrame(future, columns=['date'])
        self.future = data_prep.df_new_columns(self.df2)[['month', 'year', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']]
        self.pred = self.adabr.predict(self.future)


RandForest = RandomForest(future)
print(RandForest.feat_gini_importance.sort_values(ascending=False))
print(RandForest.results_df)
print(RandForest.pred)

mini_df = RandForest.df[['date', 'calls']]
mini_df['type'] = 'actual'
mini_df_pred = RandForest.df2
mini_df_pred['calls'] = RandForest.pred + y.reshape(365,)
mini_df_pred['type'] = 'prediction'
comp = pd.concat([mini_df, mini_df_pred], ignore_index=True)
print(comp.tail(10))

data_prep.scat_act_pred(comp)








# # Random Forest um die wichtigsten Features zu finden
# full_pred_df, feat_gini_importance, results_df = data_prep.my_model_options(df_demand_predict)
# # drücke sortierte Series 'feat' nach ihren Werten aus
# #print(feat_gini_importance.sort_values(ascending=False))
# #print(mse, r2)

# print(feat_gini_importance.sort_values(ascending=False))
# print(results_df)
# data_prep.plot_train_test(full_pred_df, df_demand_predict)