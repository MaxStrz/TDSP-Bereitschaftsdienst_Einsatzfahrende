import pandas as pd
import numpy as np
import warnings
import sys
import os
import skops.io as sio

import Code.Analysis.dataPrep as data_prep
import Code.Modeling.base_models as base_models

# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

my_file_name = 'sickness_table.csv'
column_names_types = {'date': 'datetime64[ns]', 'n_sick': 'int16', 
                      'calls': 'int32', 'n_duty': 'int16', 
                      'n_sby': 'int16', 'sby_need': 'int16', 'dafted': 'int16',
                      }
kwargs = {'my_file_name':my_file_name, 
          'column_names_types':column_names_types,}

reg_class = data_prep.RegressionCallsDemand(**kwargs)
reg_class.fit_calls_demand()
reg_class.pred_calls_demand()

trend = data_prep.DataPrediction(**kwargs)
trend.fit_trend()
trend.pred_trend()
trend.detrend()
trend.my_plot()

ins = data_prep.NewCleanedData.my_data_from_csv('sickness_table.csv', 
                                        column_names_types)
print(ins.cleaning_notes)

# stop running program here
sys.exit()

df = my_featured_data.df

viz = my_data.viz_konstruktor()

my_data.Viz.demand_vs_calls()


data_prep.overview_scatter(df)

# Erstelle neue Spalten wie 'demand'
df = data_prep.new_columns(df)

# Streuungsdiagramm zwischen 'demand' und 'calls'
data_prep.demand_vs_calls(df)

# Erstelle lineares Regressionsmodell für das Verhältnis zwischen der Anzahl der Notrufe
# und der Anzahle an gebrauchten Einsatzfahrenden
calls_demand_reg, reg_score, df = data_prep.notrufe_demand_reg(df)

# Erstelle lineares Regressionsmodell für den allgemeinen Trend der Anzahl der Notrufe
df, ax, trend_reg, reg_score = data_prep.notruf_reg(df)

data_prep.no_trend_scatter(df)

# speichere df als csv und pickle
df.to_csv('Code\\Analysis\\df1.csv')
df.to_pickle('Code\\Analysis\\df1.pkl')

# Erstelle neue Features manuell
df1 = data_prep.new_features(df)
df1.to_csv('Code\\Analysis\\new_features.csv', sep=';', decimal=',')

# Kreuzvalidierung um das beste Modell zu finden
data_prep.model_cross_val(df1)

# GridSearchCV für AdaBoost um die besten Parameter zu finden
# data_prep.adaboo_gscv(df1)

# Erstelle eigene Klassenobjekte für AdaBoost
AdaBoo = base_models.AdaBoostClass(df1)

# Mache zukünftige Vorhersagen und speichere Streuungsdiagramm davon
df3 = data_prep.future_predictions(AdaBoo, trend_reg, calls_demand_reg)

# Funktion die das Datum annimmt und die Anzahl der notwendigen 
# Einsatzfahrten im Bereitschaftsdienst für diesen Tag zurückgibt
dates = ['2019-07-08', '2019-07-09', '2019-07-10']

# pandas Series aus dates. Datentyp muss datetime sein
s_sby_need = data_prep.s_sby_need(dates)

print(s_sby_need)



df2 = AdaBoo.df2

# Füge Vorhersagen der Klassenobjekte in df ein
df = pd.concat([Reg_Class.df2, RandForest.df2['randforest_pred'], AdaBoo.df2['adaboost_pred']], axis=1)

# speichere df als csv und pickle
df.to_csv('Code\\Modeling\\df2.csv', sep=';', decimal=',')
df.to_pickle('Code\\Modeling\\df2.pkl')

# Drücke Zeilen des Dataframespalten 'date', 'status' und 'prediction' aus, wo 'status' == 'prediction'
df['call prediction with of regression and adaboost'] = (df['calls_reg_pred'] + df['adaboost_pred'])
# Lade regressionsmodell mit skops
reg = sio.load('Code\\Analysis\\model_linear_reg_demand')
# Erstelle Vorhersage für die Anzahl der Einsatzfahrten
df['n_need_predicted'] = np.round(reg.predict(df['call prediction with of regression and adaboost'].array.reshape(-1, 1)))
basis_predictions = df.loc[df['status'] == 'prediction', ['date', 'status', 'call prediction with of regression and adaboost', 'n_need_predicted']]