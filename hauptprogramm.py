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

# Lade Daten aus csv und überprüfe die Qualität der Daten
df, df_build_notes, summary_list = data_prep.sickness_table_df()

data_prep.overview_scatter(df)

# Erstelle neue Spalten wie 'demand'
df = data_prep.new_columns(df)

# Streuungsdiagramm zwischen 'demand' und 'calls'
data_prep.demand_vs_calls(df)

# Erstelle lineares Regressionsmodell für das Verhältnis zwischen der Anzahl der Notrufe
# und der Anzahle an gebrauchten Einsatzfahrenden
reg, reg_score, df = data_prep.notrufe_demand_reg(df)

# Erstelle lineares Regressionsmodell für den allgemeinen Trend der Anzahl der Notrufe
df, ax, reg, reg_score = data_prep.notruf_reg(df)

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
data_prep.adaboo_gscv(df1)

# stop running program here
sys.exit()

# Erstelle eigene Klassenobjekte für lineares Regressionsmodell, Random Forest und AdaBoost
Reg_Class = base_models.TrendRegression(df1) # Erstellen eines Klassenobjekts für das Regressionsmodell
AdaBoo = base_models.AdaBoostClass(df1)
RandForest = base_models.RandomForest(df1)

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