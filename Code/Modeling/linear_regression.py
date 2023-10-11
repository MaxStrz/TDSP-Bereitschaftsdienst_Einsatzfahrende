import pandas as pd
import os

# import df_new_columns.csv
# One folder up, then into the folder
def csv_filepath2df():
    """Verwendet absoluten Pfad des Skripts und relativen Pfad zur CSV-Datei um einen DataFrame zu erstellen"""
    current_directory = os.path.dirname(__file__) # <-- absolute dir the script is in
    filepath = os.path.join(current_directory, "../Data_AandU/df_new_columns.csv") # <-- absolute dir the script is in + path to csv file
    df = pd.read_csv(filepath, index_col=0)

    return df

df = csv_filepath2df()

print(df.head())