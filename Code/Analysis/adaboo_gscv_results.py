import pandas as pd

df = pd.read_pickle('Code\\Analysis\\adaboo_gscv_df.pkl')

score_params = ['param_estimator__max_depth', 
          'param_estimator__min_samples_split', 
          'param_learning_rate', 
          'param_n_estimators', 
          'mean_test_score', 
          'std_test_score'
          ]

splits = ['split0_test_score', 'split1_test_score', 'split2_test_score', 
          'split3_test_score', 'split4_test_score', 'split5_test_score', 
          'split6_test_score'
          ]

params = ['param_estimator__max_depth', 
          'param_estimator__min_samples_split', 
          'param_learning_rate', 
          'param_n_estimators', 
          ]

print(df[params])

params = df[params]

# alle einzigartigen Werte des DataFrames
for column in params.columns:
    print(column)
    print(params[column].unique())
    print('------------------')