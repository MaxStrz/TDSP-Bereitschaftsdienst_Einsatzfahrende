import dataPrep as data_prep
import os
import pandas as pd
import featuretools as ft
import warnings
import datetime

# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

df, df_build_notes, summary_list = data_prep.my_df()

df = data_prep.df_new_columns(df) # Neue Spalten bzw. Features erstellen

reg, reg_score, df = data_prep.notrufe_demand_reg(df)

# Print array with rounded values
print(demand_pred)

# Erstelle neue Features mit Featuretools
#feature_matrix, feature_names = dataPrep.features(df)
#print('\n'.join(map(str, feature_names)))
#print(feature_matrix.head(5))

#train_test = dataPrep.get_train_test_fm(feature_matrix)