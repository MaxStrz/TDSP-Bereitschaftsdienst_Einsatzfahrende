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

# Configuration
class Config:
    _my_file_name = 'sickness_table.csv'
    
    _column_names_types = {'date': 'datetime64[ns]', 
                          'n_sick': 'int16', 
                          'calls': 'int32', 
                          'n_duty': 'int16', 
                          'n_sby': 'int16', 
                          'sby_need': 'int16', 
                          'dafted': 'int16',
                          }
    
    _features = ('month', 'dayofmonth', 'weekday', 'weekofyear', 
                 'dayofyear', 'season')
    
    dict_config = {'my_file_name':_my_file_name, 
                   'column_names_types':_column_names_types,
                   'features':_features,
                   }

# Arrays für die Modelle
class TransformedDataArrays:
    def __init__(self,
                 df: pd.DataFrame,
                 test_days: int=47
                 ) -> None:
        self.arr_calls = np.array(df['calls']).reshape(-1, 1)
        self.arr_day = np.array(df['day']).reshape(-1, 1)
        self.arr_day_test = np.array(df['day'][-test_days:]).reshape(-1, 1)

class TrainValTestData:

    def __init__(self, 
                 df_features: pd.DataFrame,
                 s_target: pd.Series,
                 features: tuple(str),
                 test_days: int=47,
                 ) -> None:
        
        self.X = df_features[list(features)]
        self.y = s_target
        self.X_train_val = self.X[:-test_days]
        self.y_train_val = self.y[:-test_days]
        self.X_test = self.X[-test_days:]
        self.y_test = self.y[-test_days:]

# Modelle

class RegressionCallsDemand:

    def __init__(self, 
                 df_calls_demand: pd.DataFrame
                 ) -> None:
        calls = np.array(df_calls_demand['calls']).reshape(-1, 1)
        sby_needed = df_calls_demand.query('demand > 0')
        X = np.array(sby_needed['calls']).reshape(-1, 1)
        y = np.array(sby_needed['demand']).reshape(-1, 1)
        self.model_ftd_reg_calls_demand = model_ftd = self.fit(X, y)
        self.mode_scr_reg_calls_demand = self.score(model_ftd, X, y)
        self.arr_pred_demand = self.pred_calls_demand(model_ftd, calls)
        self.save_model_skops('model_linear_reg_demand.skops', 
                              model_ftd,
                              path_to_models_folder)

        
    @staticmethod    
    def fit(X: np.ndarray,
            y: np.ndarray
            ) -> LinearRegression:
        model_fitted = LinearRegression().fit(X, y)        
        return model_fitted
    
    @staticmethod
    def score(model: LinearRegression,
              X: np.ndarray,
              y: np.ndarray
              ) -> np.float64:
        model_score = model.score(X, y)
        return model_score

    @staticmethod
    def pred_calls_demand(fitted_model: LinearRegression,
                          X: np.ndarray
                          ) -> np.ndarray:
        pred_demand = fitted_model.predict(X)
        pred_demand = np.round(pred_demand, 0).astype(int)
        return pred_demand
    
    @staticmethod
    def save_model_skops(file_name: str,
                         model,
                         folder_path: str=cd
                         ) -> None:
        
        file_path = os.path.join(folder_path, file_name)
        sio.dump(model, file_path)

class DataTrend:
    def __init__(self,
                df: pd.DataFrame,
                ) -> None:
        self.arrays = TransformedDataArrays(df)
        X = self.arrays.arr_day
        y = self.arrays.arr_calls
        self.model_ftd = LinearRegression().fit(X, y)
        self.trend_score = self.model_ftd.score(X, y)
        self.save_model = self.save_model_skops('model_linear_reg_trend.skops',
                                                self.model_ftd,
                                                path_to_models_folder)
        self.calls_reg_pred = self.pred_trend(self.model_ftd, X)
        self.calls_reg_act_diff = self.detrend(df, self.calls_reg_pred)
    
    @staticmethod
    def save_model_skops(file_name: str,
                         model,
                         folder_path: str=cd
                         ) -> None:
        
        file_path = os.path.join(folder_path, file_name)
        sio.dump(model, file_path)
    
    @staticmethod
    def pred_trend(model_ftd: LinearRegression,
                   X: np.ndarray
                   ) -> np.ndarray:
        pred_calls = model_ftd.predict(X)
        pred_calls = np.round(pred_calls, 0).astype(int).reshape(-1)

        return pred_calls
    
    @staticmethod
    def detrend(df: pd.DataFrame,
                calls_reg_pred: np.ndarray
                ) -> np.ndarray:
        calls_reg_act_diff = df['calls'] - calls_reg_pred
        return calls_reg_act_diff

class AdaBooReg:
    def __init__(self, 
                 X_train, 
                 X_test, 
                 y_train, 
                 y_test) -> None:
        """Klasse für AdaBoost Vorhersagen"""
        self.X_test = X_test
        self.params_dict = params = {"adabooreg":
                                     {"n_estimators":130, 
                                      'learning_rate':0.36},
                                      "dtreg":
                                      {"criterion":'squared_error',
                                       "splitter":"best",#
                                       "max_depth":5,
                                       "min_samples_split":3}
                                       }
        self.adabooreg_mit_params = mit_params = self.adabooreg_model(params)
        self.adabooreg_ftd = model_ftd = mit_params.fit(X_train, y_train)
        self.adabooreg_pred_train = np.round(model_ftd.predict(X_train), 
                                             0).astype(int)
        self.adabooreg_pred = y_test_pred = np.round(model_ftd.predict(X_test),
                                                     0).astype(int)
        self.adabooreg_pred_all = np.concatenate((self.adabooreg_pred_train,
                                                  self.adabooreg_pred))
        self.my_metrics = self.metrics_calc(y_test, y_test_pred)
        self.gini_importance = pd.Series(model_ftd.feature_importances_,
                                         index=X_train.columns)
        
    def adabooreg_model(self,
                        params: dict[str, dict]
                        ) -> AdaBoostRegressor:
        estimator = DecisionTreeRegressor(**params['dtreg'])

        adabr = AdaBoostRegressor(estimator=estimator,
                                  random_state=42,
                                  **params['adabooreg']
                                  )
        return adabr
    
    def metrics_calc(self,
                     y_test: np.ndarray,
                     adabooreg_pred: np.ndarray
                     ) -> dict[str, np.float64]:
    
        mse = mean_squared_error(y_test, adabooreg_pred)
        r2 = r2_score(y_test, adabooreg_pred)
        metrics = {'mse':mse, 'r2':r2}
        return metrics

# Data 
class _KwargsNotUsed:
    def __init__(self, **kwargs) -> None:
        self.not_used_kwargs = kwargs

class _RawDataPath(_KwargsNotUsed):
    def __init__(self, 
                 my_file_name: str, 
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.file_path = os.path.join(path_to_raw_folder, my_file_name)
        print(f"Pfad zur CSV-Datei: {my_file_name} erstellt.")

class CleanedData(_RawDataPath):
    def __init__(self, 
                 column_names_types: dict[str, str], 
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)
        self.df_build_notes = []
        self.column_names_types = column_names_types
        self.column_names = column_names_types.keys()
        self.column_types = column_names_types.values()
        self.df_from_csv = self._make_df(self.file_path, self.column_names)
        self._missing(self.df_from_csv)
        self._is_whole_int(self.df_from_csv, column_names_types)
        self.df_cleaned = self.df_from_csv.astype(column_names_types)
        self._check_sby_duty_values(self.df_cleaned)
        self.summary_list = self._df_summary(self.df_cleaned)
        print("Daten bereinigt und in DataFrame umgewandelt.")

    def _make_df(self,
                file_path: str,
                column_names: list[str],
                date_column_name: str='date'
                ) -> pd.DataFrame:
        
        df = pd.read_csv(file_path,
                         index_col=0,
                         parse_dates=[date_column_name])
        
        # ueberpruefe ob die Liste der Spaltennamen richtig ist
        enote = "Spaltennamen sind nicht korrekt" 
        assert list(df.columns).sort() == list(column_names).sort(), enote
            
        note = "Erfolgreich: Daten erfolgreich in einen DataFrame umgewandelt"
        self.df_build_notes.append(note)

        return df

    def _missing(self, df: pd.DataFrame) -> None:
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
                        self.df_build_notes.append(note)
                    else:
                        continue
            # Drücke die nicht vollständige Liste df_build_notes aus
            [print(notes) for notes in self.df_build_notes]
        
            raise ValueError("Fehlende Daten in der CSV-Datei")
        else:
            note = "Erfolgreich: Keine fehlenden Daten in der CSV-Datei"
            self.df_build_notes.append(note)

    def _is_whole_int(self,
                       df: pd.DataFrame,
                       column_names_types: dict[str, str]
                       ) -> None:

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
            self.df_build_notes.append(note)

    def _missing_dates(self, 
                        df: pd.DataFrame,
                        date_column_name: str='date'
                        ) -> None:
        
        """Überprüft, ob alle Daten zwischen Start- und Enddatum vorhanden sind"""
        start_date = df[date_column_name].min()
        end_date = df[date_column_name].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        missing_dates = []
        for date in date_range:
            if date not in df[date_column_name].values:
                missing_dates.append(date)
        if len(missing_dates) == 0:
            note = "Erfolgreich: Alle Daten zwischen Start- und Enddatum vorhanden"
            self.df_build_notes.append(note)
        else:
            print(f"Es fehlen {len(missing_dates)} "
                  f"Daten zwischen Start- und Enddatum")
            print(f"Die fehlenden Daten sind: {missing_dates}")
            raise ValueError("Fehlende Daten zwischen Start- und Enddatum")

    def _check_sby_duty_values(self, df) -> None:
        n_sby = df['n_sby'].unique()
        self.df_build_notes.append(f"Werte in der n_sby-Spalte:{n_sby}")

        n_duty = df['n_duty'].unique()
        self.df_build_notes.append(f"Werte in der n_duty-Spalte:{n_duty}")

    def _df_summary(self, df: pd.DataFrame) -> None:
            """Gibt eine Zusammenfassung des DataFrames aus"""

            df.describe().to_csv(f"{cd}\\df_description.csv",
                                      sep=';', decimal=',')
            summary_list = []
            summary_list.append("\nDataframe Info:")
            summary_list.append(df.describe())
            summary_list.append(df.head())
            summary_list.append(df.tail())

            note = "Erfolgreich: Zusammenfassung des DataFrames als summary_list erstellt"
            self.df_build_notes.append(note)

            return summary_list

class TransformedData(CleanedData):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.startdate = sd = np.datetime64('2016-04-01')
        self.df_tformed = self.df_cleaned.copy()
        self.df_tformed['day'] = (self.df_cleaned['date']-sd).dt.days + 1
        self.df_tformed = self.create_demand(self.df_tformed)
        self.df_tformed = self.n_sick_adjusted(self.df_tformed)
        self.df_tformed = self.predict_past_demand(self.df_tformed)
        self.trend_reg = t = DataTrend(self.df_tformed)
        self.df_tformed['calls_reg_pred'] = t.calls_reg_pred
        self.df_tformed['calls_reg_act_diff'] = t.calls_reg_act_diff
        print("Daten transformiert")

    def create_demand(self, df: pd.DataFrame) -> pd.DataFrame:
        df['demand'] = np.where(df['sby_need'] > 0,
                                df['sby_need'] + df['n_duty'] -
                                df['n_sick'], np.nan
                                )
        note = "Erfolgreich: Spalte 'demand' erstellt"
        self.df_build_notes.append(note)

        return df
    
    def n_sick_adjusted(self, df: pd.DataFrame) -> pd.DataFrame:

        n_duty_1700 = np.where(df['n_duty']==1700, 
                               df['n_sick']*(19/17),
                               df['n_sick']
                               )
        n_duty_adj = np.where(df['n_duty']==1800,
                              df['n_sick']*(19/18),
                              n_duty_1700)
        df['n_sick_adj'] = np.round(n_duty_adj).astype(int)
        note = "Erfolgreich: Spalte 'n_sick_adj' erstellt"
        self.df_build_notes.append(note)

        return df

    def predict_past_demand(self,
                            df: pd.DataFrame
                            ) -> pd.DataFrame:
        df['demand_pred'] = RegressionCallsDemand(df).arr_pred_demand
        
        note = "Erfolgreich: Spalte 'demand_pred' erstellt"
        self.df_build_notes.append(note)

        return df

class FeaturedData(TransformedData):
    def __init__(self, 
                 features: tuple[str, ...], 
                 **kwargs) -> None:
        super().__init__(**kwargs)
        self.tup_features = features
        self.df_features = self.add_date_features(self.df_tformed,
                                                  self.tup_features)
        self.df_only_features = self.df_features[list(self.tup_features)]
        print('Features hinzugefügt')
    
    def add_date_features(self,
                             df: pd.DataFrame,
                             features: tuple(str),
                             date_column_name: str='date',
                             ) -> None:
        df_features = df.copy()
        c = df_features[date_column_name]

        if 'month' in features:
            df_features['month'] = c.dt.month # Monat als Zahl (1-12)

        if 'year' in features:
            df_features['year'] = c.dt.year # Jahr (4-stellig)
        
        if 'dayofmonth' in features:
            df_features['dayofmonth'] = c.dt.day # Tag des Monats (1-31)

        if 'weekday' in features:
            # Wochentag als Zahl (Montag = 0, Sonntag = 6)
            df_features['weekday'] = c.dt.weekday

        if 'weekofyear' in features:
            # Kalenderwoche als Zahl (1-52)
            df_features['weekofyear'] = c.dt.isocalendar().week

        if 'dayofyear' in features:
            # Tag des Jahres als Zahl (1-365)
            df_features['dayofyear'] = c.dt.dayofyear
        
        # Datum des 15. des vorherigen Monats
        df_features['predict_day'] = c - DateOffset(months=1, day=15)

        # Anzahl der Tage seit dem ersten Tag im Datensatz
        df_features['day'] = (c - pd.Timestamp('2016-04-01')).dt.days + 1

        if 'season' in features:
            m = df_features['month']
            # Jahreszeit als Zahl (1-4) (1=Winter, 2=Frühling, 3=Sommer, 4=Herbst)
            df_features['season'] = (m-1) % 12 // 3 + 1

        if 'calls' in df_features:
            df_features['status'] = 'actual'
        else:
            df_features['status'] = 'prediction'
        
        cols = ", ".join([str(col) for col in list(features)])
        note = f"Erfolgreich: Features {cols} hinzugefügt"
        self.df_build_notes.append(note)

        return df_features

# Vorhersagen
class DataPrediction(FeaturedData):

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.df_pred = self.df_features.copy()
        target = self.df_pred['calls_reg_act_diff']
        self.train_val_test = TrainValTestData(self.df_only_features,
                                               s_target=target,
                                               test_days=47,
                                               features=self.tup_features,
                                               )
        self.model_adabooreg = mab = self._make_adabooreg(self.train_val_test)
        self.pred_reg_calls = self._reg_pred().reshape(-1)
        self.final_ada_pred = mab.adabooreg_pred_all + self.pred_reg_calls
        self.df_pred['adabooreg_reg_calls_pred'] = self.final_ada_pred
        
    @staticmethod
    def _make_adabooreg(data: TrainValTestData) -> AdaBooReg:
        model = AdaBooReg(data.X_train_val, 
                          data.X_test, 
                          data.y_train_val,
                          data.y_test)
        return model

    def _reg_pred(self):
        self.arrays = TransformedDataArrays(self.df_features)
        arr_pred_calls = self.trend_reg.model_ftd.predict(self.arrays.arr_day)
        return arr_pred_calls

    @staticmethod
    def _date_list_in_reg_out(date: list['datetime64'],
                                startdate: 'datetime64'
                                ) -> np.array:

        # Regression für Notrufe
        skop_reg = f'{cd}\\..\\Modeling\\model_linear_reg.scops'
        reg_trend = sio.load(skop_reg, trusted=True)
        x = np.array((date-startdate).astype('timedelta64[D]') + \
                    np.timedelta64(1, 'D'))

        # numpy array aus series date
        x = x.reshape(-1, 1).astype(int)
        calls_reg_pred = reg_trend.predict(x).reshape(-1)

        return calls_reg_pred

    def df_predict_sby_need(self,
                   date: list[str]
                   ) -> pd.Series: 

        date = [np.datetime64(date) for date in date]

        calls_reg_pred = DataPrediction._date_list_in_reg_out(date, 
                                                              self.startdate)

        # Vorhersage des AdaBoost-Regressors
        #skop_adaboost = f'{cd}\\..\\Modeling\\model_adaboostreg.skops'
        #model_adaboostreg = sio.load(skop_adaboost, trusted=True)
        df_date = pd.DataFrame(date, columns=['date'])
        df_pred = self.add_date_features(df_date, self.tup_features)
        df_pred_2 = df_pred[list(self.tup_features)]
        adabooreg_pred = self.model_adabooreg.adabooreg_ftd.predict(df_pred_2).reshape(-1)

        # gesamte Vorhersage von Notrufen
        pred = calls_reg_pred + adabooreg_pred

        # Vorhersage der Nachfrage an Einsatzfahrenden
        skop_demand = f'{cd}\\model_linear_reg_demand.skops'
        model_demand = sio.load(skop_demand, trusted=True)
        sby_plus_duty = model_demand.predict(pred.reshape(-1, 1))

        # Subtrahieren von n_duty, vorausgestetzt n_duty ist 1900
        sby_need = np.round(((sby_plus_duty - 1900).reshape(-1)), 0).astype(int)

        sby_need_series = pd.Series(sby_need, index=date, name='sby_need_pred')

        s_sby_need = pd.Series(sby_need, name='sby_need_pred')
        s_adabooreg_pred = pd.Series(adabooreg_pred, name='adabooreg_pred')
        s_calls_reg_pred = pd.Series(calls_reg_pred, name='calls_reg_pred')

        df_pred = pd.concat([df_pred['date'], s_sby_need], axis=1)

        return df_pred

# Visualisierungen
class VizualisedData:
    
    def __init__(self, df: pd.DataFrame) -> None:
        self.df = df

    def demand_vs_calls(self) -> None:
        # Erstelle ein Streuungsdiagramm mit 'sby_need' auf der x-Achse und 'calls' auf der y-Achse
        fig, ax = plt.subplots()
        ax.scatter(self.df['calls'], self.df['demand'], s=3)
        # Beschriftung der Achsen und Titel
        ax.set_title('Streuungsdiagramm: Notrufe und Gesamtnachfrage')
        ax.set_xlabel('Anzahl der Notrufe')
        ax.set_ylabel('Gesamtnachfrage')
        # Zeige das Diagramm
        plt.savefig(f'{cd}\\demand_vs_calls.jpg')

    def overview_scatter(self) -> None:

        # Erstelle das Streuungsdiagramm erneut mit modifizierten Beschriftungen für die x-Achse
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
        ax1.scatter(self.df['date'], self.df['calls'], s=3)
        ax2.scatter(self.df['date'], self.df['n_sick'], s=3)
        ax3.scatter(self.df['date'], self.df['sby_need'], s=3)
        ax4.scatter(self.df['date'], self.df['dafted'], s=3)
        ax5.scatter(self.df['date'], self.df['n_duty'], s=3)

        # Hauptmarkierung der x-Achse Monatsnamen in abgekürzter Form
        ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        # Hauptmarkierung der x-Achse mit Interval von 3 Monaten
        ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        # Nebenmarkierung der x-Achse als Jahr
        ax5.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
        # Nebenmarkierung für jedes Jahr
        ax5.xaxis.set_minor_locator(mdates.YearLocator())

        # Abstand der Jahr-Beschriftungen vom Plot vergrößen
        for ax in range(4):
            fig.axes[ax].tick_params(axis='x', which='both', length=0)
    
        ax5.tick_params(axis='x', which='minor', length=10, pad=25)
        ax5.tick_params(axis='x', which='major', length=5)

        # Beschriftung der Achsen und Titel
        title = 'Streuungsdiagramm: Notrufe und Krankmeldungen nach Datum'
        ax1.set_title(title)
        ax1.set_ylabel('Anzahl der Notrufe')
        ax2.set_ylabel('Krank gemeldet')
        ax3.set_ylabel('Bereitschaftsdienst \naktiviert')
        ax4.set_ylabel('Zusätzliche \nEinsatzfahrende \naktiviert')
        ax5.set_ylabel('Einsatzfahrende \nim Dienst')
        ax5.set_xlabel('Datum')

        for ax in fig.get_axes():
            # y-Achse für alle Axen in Figure mit 0 beginnen lassen
            ax.set_ylim(bottom=0)
            # y-Achse für alle Axen in Figure + 
            # 5% des höchsten Wert der jeweiligen Spalte
            ax.set_ylim(top=ax.get_ylim()[1]*1.05)

        # Speichere das Diagramm
        to_save_fig = f'{cd}\\overview_scatter.jpg'
        plt.savefig(to_save_fig)

    def plot_notruf_reg(self) -> None:
        # Streuungsdiagramm mit 'date' auf der x-Achse
        # und 'calls' als Punkte und 'calls_pred' als Linie
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.df['date'], self.df['calls'], s=3)
        ax.plot(self.df['date'], self.df['calls_reg_pred'], color='red')
        plt.savefig(f'{cd}\\notruf_reg.jpg')

    def no_trend_scatter(self) -> None:
        # Erstelle das Streuungsdiagramm erneut mit modifizierten Beschriftungen für die x-Achse
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 10))
        date = self.df['date']
        calls = self.df['calls']
        calls_reg_act_diff = self.df['calls_reg_act_diff']
        calls_reg_pred = self.df['calls_reg_pred']

        ax1.scatter(date, calls, s=3)
        ax1.plot(date, calls_reg_pred, color='red')
        ax2.scatter(date, calls_reg_act_diff, s=3)

        # Hauptmarkierung der x-Achse Monatsnamen in abgekürzter Form
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
        # Hauptmarkierung der x-Achse mit Interval von 3 Monaten
        ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        # Nebenmarkierung der x-Achse als Jahr
        ax2.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
        # Nebenmarkierung für jedes Jahr
        ax2.xaxis.set_minor_locator(mdates.YearLocator())
        
        ax1.tick_params(axis='x', which='both', length=0)
        ax1.set_yticks(np.arange(4000, 13000, 1000))
        ax2.tick_params(axis='x', which='minor', length=10, pad=25)
        ax2.tick_params(axis='x', which='major', length=5)

        # Beschriftung der Achsen und Titel
        ax1.set_title('Trends der Notrufe')
        ax1.set_ylabel('Anzahl der Notrufe')
        ax2.set_title('Stationäre Komponente der Notrufe')
        ax2.set_ylabel('Abweichung von\n Regressionsgeraden')

        for ax in fig.get_axes():
            # y-Achse für alle Axen in Figure - 5% des kleinsten Werts der jeweiligen Spalte
            ax.set_ylim(bottom=ax.get_ylim()[0]-500)
            # y-Achse für alle Axen in Figure + 5% des höchsten Werts der jeweiligen Spalte
            ax.set_ylim(top=ax.get_ylim()[1]+500)

        # Speichere das Diagramm
        to_save_fig = f'{cd}\\no_trend_scatter.jpg'
        plt.savefig(to_save_fig)

    def actual_vs_pred(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(self.df['date'], 
                   self.df['demand_pred'], 
                   s=3, 
                   color='blue')
        
        ax.scatter(self.df['date'], 
                   self.df['nachfrage_pred'],
                   s=3, 
                   color='red')

        ax.set_title('Actual vs. Predicted Demand')
        ax.set_xlabel('Date')
        ax.set_ylabel('Demand')

        plt.savefig(f'{cd}\\demand_vs_adaboo.jpg')
    
    def scat_act_pred(self):
        # Erstelle ein Streuungsdiagramm mit 'date' auf der 
        # x-Achse und 'calls' auf der y-Achse
        fig, ax = plt.subplots()
        # Aus der 'type'-Spalte sind 'Actual'-Daten blau und 
        # 'Prediction'-Daten rot
        ax.scatter(self.df['date'], 
                   self.df['calls'], 
                   s=3, 
                   color=self.df['type'].map({'actual':'blue', 
                                              'prediction':'red'}))
        # Achsenbeschriftung
        ax.set_xlabel('Datum')
        ax.set_ylabel('Anzahl der Notrufe')
        # Zeige das Diagramm
        plt.show()

    def plot_train_test(self):
        fig, (ax_1, ax_2, ax_3) = plt.subplots(3, 
                                               1, 
                                               sharex=True, 
                                               figsize=(10, 8))
        date = self.df['date']
        y_1 = self.df['calls_reg_act_diff']
        y_2 = self.df['randforest_pred']
        y_3 = self.df['adaboost_pred']
        # Beide y-Achsen in einem figure als Streuungsdiagramme mit kleinen Punkten
        ax_1.scatter(date, y_1, color='red', marker='.')
        ax_2.scatter(date, y_2, color='blue', marker='.')
        ax_3.scatter(date, y_3, color='green', marker='.')
        # Achsenbeschriftung
        ax_1.set_xlabel('Datum')
        ax_1.set_ylabel('Calls_reg_act_diff')
        ax_2.set_ylabel('randforest_pred')
        ax_3.set_ylabel('adaboost_pred')
        
        ax_1.set_title('Calls_reg_act_diff und Predictions')
        
        plt.show()

# Kreuzvalidierung bzw. Cross Validation bzw. CV

class CrossValidatedModels:

    def __init__(self) -> None:
        kwargs = Config.dict_config
        self.cls_featured_data = cls_fd = FeaturedData(**kwargs)
        y = cls_fd.df_features['calls_reg_act_diff']
        X = cls_fd.df_only_features
        self.dict_params = dict_params = self._make_dict_params()
        self.scoring = scoring = self._set_score_types()
        tup_models = CrossValidatedModels._make_models(dict_params)
        self.model_names = CrossValidatedModels._model_names(tup_models)
        self.tscv = tscv = self._make_time_series_cv()
        self.df_model_performance = self._model_cross_val(X, y, 
                                                         tup_models, scoring,
                                                         tscv)
   
    def _make_time_series_cv(self) -> TimeSeriesSplit:
        tscv = TimeSeriesSplit(n_splits=7, test_size=32, gap=15)
        return tscv
    
    def _make_dict_params(self) -> dict[str, dict]:
        dict_params = {
            "rf": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
            "adabr": {"n_estimators": 100, "learning_rate": 1, 
                      "random_state": 42},
            "gradb": {"n_estimators": 100, "learning_rate": 1, 
                      "random_state": 42},
        }
        return dict_params
    
    def _set_score_types(self) -> list[str]:
        scoring = ['neg_mean_squared_error', 'neg_root_mean_squared_error',
                   'neg_mean_absolute_error', 'r2']
        return scoring
    
    @staticmethod
    def _model_names(tup_models) -> list[str]:
        model_names = []
        for model in tup_models:
            model_names.append(model.__class__.__name__)
        return model_names

    @staticmethod
    def _make_models(dict_params: dict[str, dict]) -> tuple:

        # Random Forest Regressor erstellen
        rf = RandomForestRegressor(**dict_params['rf'])
        adabr = AdaBoostRegressor(**dict_params['adabr']) # Teil des Basismodells
        gradb = GradientBoostingRegressor(**dict_params['gradb'])

        models = (rf, adabr, gradb)

        return models

    @staticmethod
    def _model_cross_val(X: pd.DataFrame,
                        y: pd.Series,
                        models: tuple[sklearn.base.BaseEstimator],
                        scoring: list[str],
                        tscv: TimeSeriesSplit
                        ) -> pd.DataFrame:

        # Train/Test TimerSeriesSplit um ungefähr ein Jahr abzudecken
        # Split funktioniert nicht genau wie in der eckten Welt
        # test_size ist die Anzahl der Beobachtungen in jedem 
        # Validierungsset.
        # gap ist die Anzahl an Beobachtungen zwischen dem Trainings-
        # und Validierungsset
        # gap maximal 30 - 15 sein
        # test_size mindestens 1 (31st) + 31 (Tage in langem Monat) sein

        df_model_performance = pd.DataFrame()
        
        for model in models:
            means, stds = CrossValidatedModels._cross_val(model, 
                                                          X, 
                                                          y, 
                                                          tscv, 
                                                          scoring)
            
            df_model_performance = pd.concat([df_model_performance, 
                                              means, 
                                              stds], 
                                              axis=1)
        
        return df_model_performance

    @staticmethod
    def _cross_val(model, X, y, cv, scoring):
        results = cross_validate(model, 
                                 X,
                                 y,
                                 return_estimator=True,
                                 cv=cv,
                                 scoring=scoring
                                 )
        
        mse = -results['test_neg_mean_squared_error']
        rmse = -results['test_neg_root_mean_squared_error']
        mae = -results['test_neg_mean_absolute_error']
        r2 = results['test_r2']

        # mean to 2 decimal places
        msem = mse.mean().round(decimals=2)
        rmsem = rmse.mean().round(decimals=2)
        maem = mae.mean().round(decimals=2)
        r2m = r2.mean().round(decimals=2)

        # std to 2 decimal places
        mse_std = mse.std().round(decimals=2)
        rmse_std = rmse.std().round(decimals=2)
        mae_std = mae.std().round(decimals=2)
        r2_std = r2.std().round(decimals=2)

        model_name = model.__class__.__name__

        # define series index 
        index = ['mse', 'rmse', 'mae', 'r2']

        means = np.array([msem, rmsem, maem, r2m])
        std = np.array([mse_std, rmse_std, mae_std, r2_std])

        means = pd.Series(means, name=f'{model_name}_mean', 
                          index=index, dtype='float64')

        stds = pd.Series(std, name=f'{model_name}_std',
                         index=index, dtype='float64')
        
        return means, stds

# Grid Search Cross Validation bzw. GSCV

class GSCVModels:
    def __init__(self) -> None:
        kwargs = Config.dict_config
        self.cls_featured_data = cls_fd = FeaturedData(**kwargs)
        y = cls_fd.df_features['calls_reg_act_diff']
        X = cls_fd.df_only_features
        self.dict_params = dict_params = self._make_dict_params()
        self.scoring = scoring = self._set_score_types()
        tup_models = CrossValidatedModels._make_models(dict_params)
        self.model_names = CrossValidatedModels._model_names(tup_models)
        self.tscv = tscv = self._make_time_series_cv()
        self.df_model_performance = self._model_cross_val(X, y, 
                                                        tup_models, scoring,
                                                        tscv)

    def adaboo_gscv(df):

        # Merkmalsvariablen von Zielvariable trennen
        X = df[['month', 'dayofmonth', 'weekday', 'weekofyear', 'dayofyear', 'season']]
        y = df['calls_reg_act_diff']

        # Train/Test TimerSeriesSplit
        tss = TimeSeriesSplit(n_splits=7, test_size=32, gap=15)

        d_tree_crit = ['squared_error', 'friedman_mse', 
                    'absolute_error']
        
        # Parameter für GridSearchCV
        param_grid = {"n_estimators":[135, 140],
                    #'learning_rate':[0.36, 0.35, 0.34],
                    #"estimator__criterion":d_tree_crit,
                    #"estimator__splitter":"best",
                    #"estimator__max_depth":[2, 3, 4, 8],
                    #"estimator__min_samples_split":[2, 5]
                    }

        # adaboost regressor erstellen
        adabr = AdaBoostRegressor(estimator=DecisionTreeRegressor(), 
                                random_state=42)
        
        adaboo_gscv = GridSearchCV(adabr, param_grid=param_grid, 
                        scoring='neg_mean_squared_error', cv=tss)
        
        adaboo_gscv.fit(X, y)

        # sort dataframe by rank_test_score
        adaboo_gscv_df = pd.DataFrame(adaboo_gscv.cv_results_)
        adaboo_gscv_df = adaboo_gscv_df.sort_values(by='rank_test_score')

        df_to_append = pd.read_pickle('Code\\Analysis\\adaboo_gscv_df.pkl')
        appended = pd.concat([df_to_append, adaboo_gscv_df], axis=0, sort=False)
        appended.sort_values(by=['mean_test_score'], ascending=False)

        # speichere das DataFrame als pickle
        appended.to_pickle(f'{current_directory}\\adaboo_gscv_df.pkl')

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

# ALternative Klassenstruktur

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
    startdate = np.datetime64('2016-04-01')

    def __init__(self, df_cleaned: pd.DataFrame) -> None:
        
        self._df_cleaned = df_cleaned
        self.df_transformed = self._transform_cleaned_df()

    def _transform_cleaned_df(self) -> pd.DataFrame:
        s_demand = self.create_demand()
        s_n_sick_adj = self.n_sick_adjusted()
        df_transformed = self._df_cleaned.assign(demand=s_demand, 
                                                 n_sick_adj=s_n_sick_adj)
        return df_transformed

    def create_demand(self) -> pd.Series:
        df = self._df_cleaned # Alias für _df_cleaned
        s_demand = np.where(df['sby_need'] > 0,
                            df['sby_need'] + df['n_duty'] -
                            df['n_sick'], np.nan)
        return s_demand
        
    def n_sick_adjusted(self) -> pd.Series:
        df = self._df_cleaned # Alias für _df_cleaned
        n_duty_1700 = np.where(df['n_duty']==1700, 
                               df['n_sick']*(19/17),
                               df['n_sick'])
        n_duty_adj = np.where(df['n_duty']==1800,
                              df['n_sick']*(19/18),
                              n_duty_1700)
        s_n_sick_adj = np.round(n_duty_adj).astype(int)
        return s_n_sick_adj

    # day = np.array((self.df['date']-self.startdate).dt.days + 1)
    # self.arr_day = day.reshape(-1, 1)
    # self.arr_calls = np.array(self.df['calls']).reshape(-1, 1)

# archiv

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
