import pandas as pd
import warnings
import matplotlib.pyplot as plt

import dataPrep as data_prep

# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

df, df_build_notes, summary_list = data_prep.my_df()

df_2 = data_prep.df_new_columns(df)

df_2 = df_2.drop(['n_duty', 'n_sby', 'sby_need', 'dafted', 'predict_day'], axis=1)

reg, reg_score, df_demand_predict = data_prep.notrufe_demand_reg(df_2)

df_demand_predict, ax = data_prep.notruf_reg(df_demand_predict)

# speichere df_2 als csv
df_demand_predict.to_csv('df_new_columns.csv')
df_demand_predict.to_pickle('df_new_columns.pkl')

# df_2 = data_prep.df_new_columns(df) # Neue Spalten erstellen
# df_2 = df_2.drop(['n_duty', 'n_sby', 'sby_need', 'dafted', 'predict_day'], axis=1)
# reg, reg_score, df_demand_predict = data_prep.notrufe_demand_reg(df_2)
# df_demand_predict, ax = data_prep.notruf_reg(df_demand_predict)
# df_demand_predict = df_demand_predict.drop(['calls','n_sick', 'demand', 'demand_pred', 'calls_pred'], axis=1)
# # Random Forest um die wichtigsten Features zu finden
# full_pred_df, feat_gini_importance, results_df = data_prep.my_model_options(df_demand_predict)
# # drücke sortierte Series 'feat' nach ihren Werten aus
# #print(feat_gini_importance.sort_values(ascending=False))
# #print(mse, r2)

# print(feat_gini_importance.sort_values(ascending=False))
# print(results_df)
# data_prep.plot_train_test(full_pred_df, df_demand_predict)

# df_demand_predict = df_demand_predict.drop(['season', 'year', 'month'], axis=1)
# full_pred_df, feat_gini_importance, results_df = data_prep.my_model_options(df_demand_predict)
# data_prep.plot_train_test(full_pred_df, df_demand_predict)

# print(feat_gini_importance.sort_values(ascending=False))
# print(results_df)


# #plt.show()