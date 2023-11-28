from __future__ import annotations
import featuretools as ft
from featuretools.primitives import Lag, RollingMin, RollingMean
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np  
import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
import sklearn
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
import skops.io as sio
from statsmodels.tsa.seasonal import seasonal_decompose
import sys
import warnings

# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

# absolute dir in dem das Skript ist
current_directory = os.path.dirname(__file__)
cd = os.path.dirname(__file__) # absolute dir in dem das Skript ist
relative_path_to_raw_folder = "../../Sample_Data/Raw/"
relative_path_to_models_folder = "../Modeling"
path_to_raw_folder = os.path.join(cd, relative_path_to_raw_folder)
path_to_models_folder = os.path.join(cd, relative_path_to_models_folder)

# Alternative Klassenstruktur

class NewCleanedData:

    def __init__(self, df: pd.DataFrame, cleaning_notes: list[str]) -> None:
        self.df = df
        self.cleaning_notes = cleaning_notes
    
    @classmethod
    def my_data_from_csv(cls, 
                         raw_file_name: str, 
                         column_names_types: dict[str, str]
                         ) -> MyData:
        
        my_raw_file_path = _RawDataPath(raw_file_name).file_path
        df = pd.read_csv(my_raw_file_path, index_col=0, parse_dates=['date'])
        cleaning_notes = cls._quality_checks(df, column_names_types)
        df = df.astype(column_names_types)
        return cls(df, cleaning_notes)

    @classmethod
    def _quality_checks(cls, 
                        df: pd.DataFrame, 
                        column_names_types: dict[str, str]
                        ) -> list[str]:
        
        note_missing = cls._missing(df)
        note_is_whole_int = cls._is_whole_int(df, column_names_types)
        note_missing_dates = cls._missing_dates(df, column_name='date')
        note_n_sby_values = cls._check_n_sby_values(df)
        note_n_duty_values = cls._check_n_duty_values(df)
        
        cleaning_notes = [note_missing, note_is_whole_int, 
                          note_missing_dates, note_n_sby_values,
                          note_n_duty_values]

        return cleaning_notes

    @staticmethod
    def _missing(df) -> str:
        """
        Überprüft, ob es fehlende Daten in den Spalten des DataFrames 
        gibt. Wenn ja, gibt es eine ValueError-Exception mit einer 
        Liste von fehlenden Daten aus.

        Args:
            df (pandas.DataFrame): Der DataFrame, der überprüft 
            werden soll.

        Raises:
            ValueError: Wenn es fehlende Daten in der CSV-Datei gibt.

        Returns:
            None
        """
        # Überprüft ob es fehlende Daten in den jeweiligen Spalten gibt.
        # pd.Series mit Spalten als Index und Wert True wenn es 
        # fehlende Daten gibt, sonst False
        df_missing = df.isnull().any()
        if df_missing.any():
            # for-Schleife um die fehlenden Daten in der jeweiligen 
            # Spalte zu finden
            for col in df_missing.index:
                # enumerate() gibt den Index und Wert jedes Elements 
                # in der Spalte aus
                for index, value in enumerate(df[col]):
                    if pd.isna(value):
                        # Füge die Spalte, das Datum und den Index des 
                        # fehlenden Wertes in die Liste ein
                        note = (f"Nicht erfolgreich:\n"
                                f"Es fehlen Daten in Spalte: "
                                f"{col}, Index: {index}")
                    else:
                        continue
            raise ValueError("Fehlende Daten in der CSV-Datei")
        else:
            note = "Erfolgreich: Keine fehlenden Daten in der CSV-Datei"
        
        return note

    @staticmethod
    def _is_whole_int(df, column_names_types) -> str:

        # Leere Liste für nicht-ganzzahlige Werte
        non_int_list = []

        # Liste alle integar-Spalten
        t = column_names_types.items()
        int_cols = [c for c, v in t if v=='int16' or v=='int32' or v=='int64']

        # Überprüft ob alle Werte in der Spalte 'calls', 'sby_need', 
        # 'dafted', 'n_sick', 'n_duty', 'n_sby' gleich ihrem 
        # Interger-Wert sind. Wenn nicht, raise error und gebe das Datum
        #  aus der 'date'-Spalte und Index des fehlerhaften Wertes aus.
        for col in int_cols:
            for index, value in enumerate(df[col]):
                # Drücke ganze Zeile von index aus
                if value != int(value):
                    # Füge die Spalte, das Datum und den Index des 
                    # fehlenden Wertes in die Liste ein
                    non_int_list.append([col, df.loc[index]])
                else:
                    continue
        
        # Wenn die Liste nicht leer ist, beschreibe die 
        # fehlerhaften Daten und raise error
        if len(non_int_list) != 0:
            print(f"Es gibt {len(non_int_list)} nicht-ganzzahlige "
                    f"Werte im Datensatz:")
            for data in non_int_list:
                print(f"Spalte: {data[0]}, Zeile: {data[1]}")

            note = f"Nicht-ganzzahlige Werte in den Spalten {int_cols}"
            raise ValueError(note)
        else:
            cols = ", ".join([str(col) for col in int_cols])
            note = f"Erfolgreich: Keine nicht-ganzzahligen Werte in den Spalten {cols}"
        
        return note

    @staticmethod
    def _missing_dates(df, column_name: str='date') -> str:
        """Überprüft, ob alle Daten zwischen Start- und Enddatum vorhanden sind"""
        start_date = df[column_name].min()
        end_date = df[column_name].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        missing_dates = []
        for date in date_range:
            if date not in df[column_name].values:
                missing_dates.append(date)
        if len(missing_dates) == 0:
            note = "Erfolgreich: Alle Daten zwischen Start- und Enddatum vorhanden"
        else:
            print(f"Es fehlen {len(missing_dates)} "
                  f"Daten zwischen Start- und Enddatum")
            print(f"Die fehlenden Daten sind: {missing_dates}")
            raise ValueError("Fehlende Daten zwischen Start- und Enddatum")
        
        return note

    @staticmethod
    def _check_n_sby_values(df) -> str:
        n_sby = df['n_sby'].unique()
        return f"Werte in der n_sby-Spalte:{n_sby}"

    @staticmethod
    def _check_n_duty_values(df) -> str:
        n_duty = df['n_duty'].unique()
        return f"Werte in der n_duty-Spalte:{n_duty}"

class NewTransformedData:

# Funktionen

def ts_split_train(df):

    # Merkmalsvariablen von Zielvariable trennen
    X = df[['month', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']]
    y = df['calls_reg_act_diff']

    # Train/Test TimerSeriesSplit
    tss = TimeSeriesSplit(n_splits=3, test_size=32, gap=15)

    # adaboost regressor erstellen
    adabr = AdaBoostRegressor(random_state=42)
        
    # Random Forest Regressor erstellen
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
    adabr = AdaBoostRegressor(n_estimators=100, random_state=42, learning_rate=1) # Teil des Basismodells
    models = (rf, adabr)

    # Leere Dictionary für die Ergebnisse
    val_dict = {}

    # Trainiere das Modell auf Trainingsdaten
    for m in models:
        val_dict[m.__class__.__name__] = []
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            m.fit(X_train, y_train)

            y_pred = m.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            val_dict[m.__class__.__name__].append(mse)
    
    print(pd.DataFrame(val_dict))

def actual_vs_pred(df3):
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(df3['date'], df3['demand_pred'], s=3, color='blue')
    ax.scatter(df3['date'], df3['nachfrage_pred'], s=3, color='red')

    ax.set_title('Actual vs. Predicted Demand')
    ax.set_xlabel('Date')
    ax.set_ylabel('Demand')

    plt.savefig(f'{current_directory}\\demand_vs_adaboo.jpg')

def future_predictions(AdaBoo, trend_reg, calls_demand_reg):

    df2 = AdaBoo.df2

    df3 = demand_pred_final(df2, trend_reg, calls_demand_reg)
 
    df3['sby_pred'] = df3['nachfrage_pred'] - df3['n_duty']

    actual_vs_pred(df3)

    return df3

def my_model_options(df):

    # Merkmalsvariablen von Zielvariable trennen
    X = df[['month', 'year', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']]
    y = df['calls_reg_act_diff']
    X_train = X[:-47]
    y_train = y[:-47]
    X_test = X[-47:]
    y_test = y[-47:]

    # Random Forest Regressor erstellen
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)

    estimator = DecisionTreeRegressor(max_depth=5, min_samples_split=3)
    adabr = AdaBoostRegressor(estimator=estimator,
                              n_estimators=130, 
                              random_state=42, 
                              learning_rate=0.36)
    models = [rf, adabr]

    rf.fit(X_train, y_train)
    adabr.fit(X_train, y_train)

    # Persist das Modell mit skops
    sio.dump(adabr, f'{cd}\\..\\Modeling\\model_adaboostreg.skops')
    sio.dump(rf, f'{cd}\\..\\Modeling\\model_randomforestreg.skops')

    # Sorted Feature Importance
    feature_gini_importance = pd.Series(rf.feature_importances_, index=X_train.columns)

    # Vorhersagen auf Testdaten
    predictions_rf = rf.predict(X_test)
    predictions_adabr = adabr.predict(X_test)

    # Series der Vorhersagen aller X-Werte
    full_pred_rf = rf.predict(X)
    full_pred_adabr = adabr.predict(X)


    # Vorhersagen der Modelle in den DataFrame einfügen
    df['randforest_pred'] = full_pred_rf
    df['adaboost_pred'] = full_pred_adabr

    # Berechne mean squared error für die Vorhersagen
    mse_rf = mean_squared_error(y_test, predictions_rf)
    r2_rf = r2_score(y_test, predictions_rf)

    mse_adabr = mean_squared_error(y_test, predictions_adabr)
    r2_adabr = r2_score(y_test, predictions_adabr)

    # Dictionary mit den Ergebnissen
    results = {'mse_rf':mse_rf, 'r2_rf':r2_rf, 'mse_adabr':mse_adabr, 'r2_adabr':r2_adabr}
    results_df = pd.DataFrame(results, index=[0])

    return df, feature_gini_importance, results_df, models

def adaboo_fut_predict(self):
    """Vorhersage der Anzahl der Notrufe"""
    df = future(self.df1)
    miss_adaboost_pred = df['adaboost_pred'].isna() # Series mit True/False, ob Wert fehlt
    features = df.loc[miss_adaboost_pred, ['month', 'year', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']] # Dataframe mit Features, an de fehlt
    adaboost_pred = self.adabr.predict(features) # Vorhersage der Anzahl der Notrufe
    df.loc[miss_adaboost_pred, 'adaboost_pred'] = adaboost_pred # Füge Vorhersage in df ein wo Wert fehlt
    return df

def features(df):

    # Entferne unwichtige Spalten
    df_feat_2 = df.drop(['n_duty', 'n_sby', 'sby_need', 'dafted', 'predict_day',
                        'month', 'demand', 'calls_pred', 'calls', 'n_sick'],
                        axis=1)
    
    cutoff_df = df.drop(['n_duty', 'n_sby', 'sby_need', 'dafted', 'calls_diff', 'season', 'date',
                     'month', 'demand', 'calls_pred', 'calls', 'n_sick'],
                    axis=1)
    
    # rename cutoff_df['predict_day'] to cutoff_df['time']
    cutoff_df = cutoff_df.rename(columns={'predict_day':'time'})
    # cutoff_df['instance_id'] = cutoff_df.index
    cutoff_df['instance_id'] = cutoff_df.index

    def train_test_daten(df):
        """
        Teilt den Datensatz in Trainings- und Testdaten mithilfe von 
        sklearn.model_selection.train_test_split auf	
        """
        df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

        df_train['test_data'] = False # Spalte 'test_data' erstellen und mit False füllen
        df_test['test_data'] = True # Spalte 'test_data' erstellen und mit True füllen

        # Zusammenführen der beiden DataFrames.
        # sort=True: Sortiert die Spalten alphabetisch nach Spaltennamen
        data = pd.concat([df_train, df_test], sort=True)
        data['index'] = data.index # Index als Spalte hinzufügen

        return data

    data = train_test_daten(df_feat_2) # Datensatz in Trainings- und Testdaten mit Spalte 'test_data' aufteilen

    es = ft.EntitySet('einsatzfahrende') # Erstelle EntitySet mit Namen 'einsatzfahrende'

    es.add_dataframe( # Füge Dataframe 'data' zum EntitySet hinzu
        data,
        dataframe_name='data',
        index='index',
        time_index='date'
        )
    
    # Lücke zwischen Zielzeitpunkt und letztem Zeitpunkt im Datensatz für den Feature-Berechnungszeitpunkt
    #gap = 45
    # Zeitfenster für den Feature-Engineering
    #window_length = 100

    # Delaying-Primitives  
    #delaying_primitives = [ft.primitives.Lag(periods=i + gap) for i in range(window_length)]

    #rolling_mean_primitive = RollingMean(window_length=window_length, 
    #                                     gap=gap, min_periods=window_length)

    #rolling_min_primitive = RollingMin(window_length=window_length, 
    #                                   gap=gap, min_periods=window_length)
    
    # Erstelle Features mit Deep Feature Synthesis
    # 'day' ist Tag des Monats (1-31)
    # 'week' ist Kalenderwoche (1-52)
    datetime_primitives = ['month', 'weekday', 'is_weekend', 'week', 'day', 'year', 'day_of_year']

    features, feature_names = ft.dfs(entityset=es, # EntitySet
                                    target_dataframe_name='data', # Ziel-Dataframe	
                                    trans_primitives=(
                                        datetime_primitives),
                                        #+ delaying_primitives
                                        #+ [rolling_mean_primitive, rolling_min_primitive]),
                                    cutoff_time=cutoff_df, # Startdatum für Feature-Berechnung
                                    )

    return features, feature_names

def train_test_daten(df):
    """
    Teilt den Datensatz in Trainings- und Testdaten mithilfe von 
    sklearn.model_selection.train_test_split auf	
    """
    df_train, df_test = train_test_split(df, test_size=0.2, random_state=42)

    df_train['test_data'] = False # Spalte 'test_data' erstellen und mit False füllen
    df_test['test_data'] = True # Spalte 'test_data' erstellen und mit True füllen

    # Zusammenführen der beiden DataFrames.
    # sort=True: Sortiert die Spalten alphabetisch nach Spaltennamen
    data = pd.concat([df_train, df_test], sort=True)
    data['index'] = data.index

    return data

def get_train_test_fm(feature_matrix):

    X_train = feature_matrix[feature_matrix['test_data'] == False]
    X_train = X_train.drop(columns=['test_data']) # drop test_data Spalte
    train_labels = X_train['calls']
    X_train = X_train.drop(columns=['calls']) # drop calls Spalte
    X_test = feature_matrix[feature_matrix['test_data'] == True]
    test_labels = X_test['calls']
    X_test = X_test.drop(columns=['test_data', 'calls']) # drop test_data und calls Spalte
    
    train_test = {'x_train':X_train, 'train_labels':train_labels, 'x_test':X_test, 'test_labels':test_labels}
    
    def df_to_csv(df, name):
        """Verwendet absoluten Pfad des Skripts und relativen Pfad zur CSV-Datei um einen DataFrame zu erstellen"""
        current_directory = os.path.dirname(__file__) # <-- absolute dir the script is in
        # Namen des Dataframes
        filepath = os.path.join(current_directory, f"../../Sample_Data/Processed/{name}.csv") # <-- absolute dir the script is in + path to csv file
        df.to_csv(filepath, index=False)

    for name, df in train_test.items():
        df_to_csv(df, name)
        
    return train_test

def seas_decomp(df):
    calls_series = df['calls']
    calls_series.index = df['date']
    result = seasonal_decompose(calls_series, model='additive', period=60)
    result.plot()
    #plt.show()

def is_whole_int(self) -> None:

    # Leere Liste für nicht-ganzzahlige Werte
    non_int_list = []

    # Liste alle integar-Spalten
    t = self.column_names_types.items()
    int_cols = [c for c, v in t if v=='int16' or v=='int32' or v=='int64']

    # Überprüft ob alle Werte in der Spalte 'calls', 'sby_need', 
    # 'dafted', 'n_sick', 'n_duty', 'n_sby' gleich ihrem 
    # Interger-Wert sind. Wenn nicht, raise error und gebe das Datum
    #  aus der 'date'-Spalte und Index des fehlerhaften Wertes aus.
    for col in int_cols:
        for index, value in enumerate(self.df[col]):
            # Drücke ganze Zeile von index aus
            if value != int(value):
                # Füge die Spalte, das Datum und den Index des 
                # fehlenden Wertes in die Liste ein
                non_int_list.append([col, self.df.loc[index]])
            else:
                continue
    
    # Wenn die Liste nicht leer ist, beschreibe die 
    # fehlerhaften Daten und raise error
    if len(non_int_list) != 0:
        print(f"Es gibt {len(non_int_list)} nicht-ganzzahlige "
                f"Werte im Datensatz:")
        for data in non_int_list:
            print(f"Spalte: {data[0]}, Zeile: {data[1]}")

        note = f"Nicht-ganzzahlige Werte in den Spalten {int_cols}"
        raise ValueError(note)
    else:
        cols = ", ".join([str(col) for col in int_cols])
        note = f"Erfolgreich: Keine nicht-ganzzahligen Werte in den Spalten {cols}"
        self.df_build_notes.append(note)

def missing_dates(self, column_name: str='date') -> None:
    """Überprüft, ob alle Daten zwischen Start- und Enddatum vorhanden sind"""
    start_date = self.df[column_name].min()
    end_date = self.df[column_name].max()
    date_range = pd.date_range(start=start_date, end=end_date)
    missing_dates = []
    for date in date_range:
        if date not in self.df[column_name].values:
            missing_dates.append(date)
    if len(missing_dates) == 0:
        note = "Erfolgreich: Alle Daten zwischen Start- und Enddatum vorhanden"
        self.df_build_notes.append(note)
    else:
        print(f"Es fehlen {len(missing_dates)} "
                f"Daten zwischen Start- und Enddatum")
        print(f"Die fehlenden Daten sind: {missing_dates}")
        raise ValueError("Fehlende Daten zwischen Start- und Enddatum")

def set_data_types(self) -> None:
    # Alle Spalten außer 'date' in Integer umwandeln
    int_cols = ['calls', 'sby_need', 'dafted', 'n_sick', 'n_duty', 'n_sby']
    for col in int_cols:
        self.df[col] = self.df[col].astype(int)

    # Bestätigung, dass alle Spalten außer 'date' in Integer 
    # umgewandelt wurden
    note = "Erfolgreich: Alle Spalten ausser 'date' in Integer umgewandelt"
    self.df_build_notes.append(note)

    # 'date'-Spalte in Datetime umwandeln
    self.df['date'] = pd.to_datetime(self.df['date'])

    # Bestätigung, dass alle Daten in der 'date'-Spalte Datetime sind
    note = "Erfolgreich: Alle Daten in der 'date'-Spalte sind Datetime-Datentyp"
    self.df_build_notes.append(note)

def missing(self) -> None:
    """
    Überprüft, ob es fehlende Daten in den Spalten des DataFrames 
    gibt. Wenn ja, gibt es eine ValueError-Exception mit einer 
    Liste von fehlenden Daten aus.

    Args:
        df (pandas.DataFrame): Der DataFrame, der überprüft 
        werden soll.

    Raises:
        ValueError: Wenn es fehlende Daten in der CSV-Datei gibt.

    Returns:
        None
    """
    # Überprüft ob es fehlende Daten in den jeweiligen Spalten gibt.
    # pd.Series mit Spalten als Index und Wert True wenn es 
    # fehlende Daten gibt, sonst False
    df_missing = self.df.isnull().any()
    if df_missing.any():
        # for-Schleife um die fehlenden Daten in der jeweiligen 
        # Spalte zu finden
        for col in df_missing.index:
            # enumerate() gibt den Index und Wert jedes Elements 
            # in der Spalte aus
            for index, value in enumerate(self.df[col]):
                if pd.isna(value):
                    # Füge die Spalte, das Datum und den Index des 
                    # fehlenden Wertes in die Liste ein
                    note = (f"Nicht erfolgreich:\n"
                            f"Es fehlen Daten in Spalte: "
                            f"{col}, Index: {index}")
                    self.df_build_notes.append(note)
                else:
                    continue
        # Drücke die nicht vollständige Liste df_build_notes aus
        [print(notes) for notes in self.df_build_notes]
    
        raise ValueError("Fehlende Daten in der CSV-Datei")
    else:
        note = "Erfolgreich: Keine fehlenden Daten in der CSV-Datei"
        self.df_build_notes.append(note)

def clean_data(self) -> None:
    self.make_df()
    self.missing()
    self.is_whole_int()
    self.missing_dates()
    self.set_data_types()
    self.check_sby_duty_values(self.dfb)

def make_df(self) -> None:
    self.df = pd.read_csv(self.file_path, 
                            index_col=0, parse_dates=['date'])
    
    # ueberpruefe ob die Liste der Spaltennamen richtig ist
    enote = "Spaltennamen sind nicht korrekt" 
    assert list(self.df.columns).sort() == list(self.column_names).sort(), enote
        
    note = "Erfolgreich: Daten erfolgreich in einen DataFrame umgewandelt"
    self.df_build_notes.append(note)

def demand_pred_final(df2, trend_reg, calls_demand_reg):

    startdate = np.datetime64('2016-04-01')
    x2 = np.array((df2['date']-startdate).dt.days + 1).reshape(-1, 1)
    trend_2 = trend_reg.predict(x2)
    calls_pred_2 = (np.round(trend_2, 0)).astype(int)
    df2['calls_reg_pred_2'] = calls_pred_2.reshape(-1)

    df2['reg_adaboost_pred'] = df2['calls_reg_pred_2'] + df2['adaboost_pred']

    # Nachfrage basierend auf reg_adaboost_pred
    x_calls = np.array(df2['reg_adaboost_pred']).reshape(-1, 1)
    nachfrage_pred = calls_demand_reg.predict(x_calls)
    df2['nachfrage_pred'] = nachfrage_pred.reshape(-1)

    return df2
