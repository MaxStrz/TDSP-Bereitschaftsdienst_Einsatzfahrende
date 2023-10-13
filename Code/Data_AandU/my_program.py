import pandas as pd
import warnings
import matplotlib.pyplot as plt

import dataPrep as data_prep

# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

df, df_build_notes, summary_list = data_prep.my_df()

df = data_prep.df_new_columns(df)


#df_2 = df.drop(['n_duty', 'n_sby', 'sby_need', 'dafted', 'predict_day'], axis=1)
#print(df_2)

reg, reg_score, df = data_prep.notrufe_demand_reg(df)

df, ax, reg = data_prep.notruf_reg(df)

# speichere df_2 als csv
df.to_csv('df.csv')
df.to_pickle('df.pkl')