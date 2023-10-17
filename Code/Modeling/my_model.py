import sys
import pandas as pd
import numpy as np

sys.path.insert(0, '../Data_AandU')
import dataPrep as data_prep

def future(df):
    last_train_date = np.datetime64(df['date'].max())
    future = pd.date_range(start=last_train_date, periods=47, freq='D', inclusive='right')
    future_series = pd.DataFrame(future, columns=['date']) # Series mit 46 Tagen in der Zukunft
    future_series = data_prep.df_new_columns(future_series) # Dataframe mit zukünftigen Features
    df = pd.concat([df, future_series], ignore_index=True) # Dataframe mit zukünftigen Features
    return df

#create class for regression
class TrendRegression:
    def __init__(self):
        """Initialize class with a regression model"""
        self.df = pd.read_pickle('df.pkl')
        self.df1, self.ax, self.reg = self.reg_fit()
        self.reg_steigerung = self.reg.coef_
        self.reg_intercept = self.reg.intercept_
        self.df2 = self.reg_fut_predict()

    def reg_fit(self):
        """Passe Regressionsmodell an"""
        df1, ax, reg = data_prep.notruf_reg(self.df)
        return df1, ax, reg

    def reg_fut_predict(self):
        """Vorhersage der Anzahl der Notrufe"""
        df = future(self.df1)
        miss_calls_reg_pred = df['calls_reg_pred'].isna() # Series mit True/False, ob Wert fehlt
        day = df.loc[miss_calls_reg_pred, 'day'] # Series mit Tagen, an denen Wert fehlt
        day = day.values.reshape(-1, 1) # Reshape zu 2D Array für Regressionsmodell
        calls_reg_pred = self.reg.predict(day) # Vorhersage der Anzahl der Notrufe
        df.loc[miss_calls_reg_pred, 'calls_reg_pred'] = calls_reg_pred # Füge Vorhersage in df ein wo Wert fehlt
        return df

Reg_Class = TrendRegression()

class RandomForest:
    def __init__(self):
        """Klasse für Random Forest"""
        self.df = pd.read_pickle('df.pkl') # Dataframe mit allen Features
        self.df1, self.feat_gini_importance, self.results_df, self.models = data_prep.my_model_options(self.df) # Dataframe mit random forest und adaboost vorhersagen
        self.adabr = self.models[1]
        self.rf = self.models[0]
        self.df2 = self.randf_fut_predict()
    
    def randf_fut_predict(self):
        """Vorhersage der Anzahl der Notrufe"""
        df = future(self.df1)
        miss_randforest_pred = df['randforest_pred'].isna() # Series mit True/False, ob Wert fehlt
        features = df.loc[miss_randforest_pred, ['month', 'year', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']] # Dataframe mit Features, an de fehlt
        randforest_pred = self.rf.predict(features) # Vorhersage der Anzahl der Notrufe
        df.loc[miss_randforest_pred, 'randforest_pred'] = randforest_pred # Füge Vorhersage in df ein wo Wert fehlt
        return df

RandForest = RandomForest()

class AdaBoostClass:
    def __init__(self):
        """Klasse für Random Forest"""
        self.df = pd.read_pickle('df.pkl') # Dataframe mit allen Features
        self.df1, self.feat_gini_importance, self.results_df, self.models = data_prep.my_model_options(self.df) # Dataframe mit random forest und adaboost vorhersagen
        self.adabr = self.models[1]
        self.df2 = self.adaboo_fut_predict()
    
    def adaboo_fut_predict(self):
        """Vorhersage der Anzahl der Notrufe"""
        df = future(self.df1)
        miss_adaboost_pred = df['adaboost_pred'].isna() # Series mit True/False, ob Wert fehlt
        features = df.loc[miss_adaboost_pred, ['month', 'year', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']] # Dataframe mit Features, an de fehlt
        adaboost_pred = self.adabr.predict(features) # Vorhersage der Anzahl der Notrufe
        df.loc[miss_adaboost_pred, 'adaboost_pred'] = adaboost_pred # Füge Vorhersage in df ein wo Wert fehlt
        return df

AdaBoo = AdaBoostClass()

# Liste-objekt aller Spalten
columns_reg = list(Reg_Class.df2.columns)
columns_randforest = list(RandForest.df2.columns)
# Liste-objekt aller Spalten, die in beiden Dataframes enthalten sind
columns_both = list(set(columns_reg) & set(columns_randforest))
# Remove 'calls_reg_pred' from columns_both
columns_both.remove('calls_reg_pred')

print(Reg_Class.df2[columns_both].equals(RandForest.df2[columns_both]))

df = pd.concat([Reg_Class.df2, RandForest.df2['randforest_pred'], AdaBoo.df2['adaboost_pred']], axis=1)

print(df)