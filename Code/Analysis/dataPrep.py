import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np  
import os
import pandas as pd
from pandas.tseries.offsets import DateOffset
import sklearn
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.tree import DecisionTreeRegressor
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings

# ignore FutureWarnings
warnings.simplefilter(action='ignore',
                      category=(FutureWarning, pd.errors.PerformanceWarning))

cd = os.path.dirname(__file__) # absolute dir the script is in
relative_path_to_folder_data_raw = "../../Data/Raw/"
relative_path_to_folder_modeling = "../Modeling"
path_to_folder_data_raw = os.path.join(cd, relative_path_to_folder_data_raw)
path_to_folder_modeling = os.path.join(cd, relative_path_to_folder_modeling)

# Konfigurationen
class Config:
    """
    Erstelle ein Dictionary mit Konfigurationen für die Daten und Features.   
    
    Attributes
    ----------
    _my_file_name : str 
        Name der CSV-Datei mit den Daten
    _column_names_types : dict[str, str]
        Dictionary mit Spaltennamen und Datentypen
    _features : tuple[str]
        Tuple mit Features für die Vorhersage
    dict_config : dict[str, any]
        Dictionary mit Konfiguration für die Daten und Features
    """

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

# Kleine Daten Transformationen
class TransformedDataArrays:
    """
    Stelle DataFrame-Spalten und Series als Arrays für Modelle bereit.

    Attributes
    ----------
    arr_calls : np.ndarray
        Array der Spalte 'calls'
    arr_day : np.ndarray
        Array der Spalte 'day' beginnend mit 1
    arr_day_test : np.ndarray
        Array der Spalte 'day' für Testdaten
    """

    def __init__(self, df: pd.DataFrame, test_days: int=47) -> None:
        """
        Parameters
        ----------
        df : DataFrame
            DataFrame mit Spalten 'calls' und 'day'
        test_days : int, optional
            Anzahl der Tage für Testdaten

        Returns
        -------
        None.

        """
        
        self.arr_calls = np.array(df['calls']).reshape(-1, 1)
        self.arr_day = np.array(df['day']).reshape(-1, 1)
        self.arr_day_test = np.array(df['day'][-test_days:]).reshape(-1, 1)

class TrainValTestData:
    """
    Stelle Daten für Trainings-, Validierungs- und Testdaten bereit.

    Attributes
    ----------
    X : pd.DataFrame
        DataFrame mit nur Features
    y : pd.Series
        Series mit nur Zielvariable 'calls_reg_act_diff'
    X_train_val : pd.DataFrame
        DataFrame mit nur Features für Trainings- und Validierungsdaten
    y_train_val : pd.Series
        Series mit nur Zielvariable 'calls_reg_act_diff' für Train und Val
    X_test : pd.DataFrame
        DataFrame mit nur Features für Testdaten
    y_test : pd.Series
        Series mit nur Zielvariable 'calls_reg_act_diff' für Testdaten
    """

    def __init__(self, 
                 df_features: pd.DataFrame,
                 s_target: pd.Series,
                 features: tuple[str],
                 test_days: int=47,
                 ) -> None:
        """
        Parameters
        ----------
        df_features : pd.DataFrame
            DataFrame mit mindestens alle Features
        s_target : pd.Series
            Series mit Zielvariable 'calls_reg_act_diff'
        features : tuple[str]
            Tuple der Features für die Vorhersage
        test_days : int, optional
            Anzahl der Tage für Testdaten

        Returns
        -------
        None.
        """

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

class DataTrend:
    def __init__(self,
                df: pd.DataFrame,
                ) -> None:
        self.arrays = TransformedDataArrays(df)
        X = self.arrays.arr_day
        y = self.arrays.arr_calls
        self.model_ftd = LinearRegression().fit(X, y)
        self.trend_score = self.model_ftd.score(X, y)
        self.calls_reg_pred = self.pred_trend(self.model_ftd, X)
        self.calls_reg_act_diff = self.detrend(df, self.calls_reg_pred)
    
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

# Daten bereinigen und transformieren und Features hinzufügen
class _KwargsNotUsed:
    def __init__(self, **kwargs) -> None:
        self.not_used_kwargs = kwargs

class _RawDataPath(_KwargsNotUsed):
    def __init__(self, 
                 my_file_name: str, 
                 **kwargs
                 ) -> None:
        super().__init__(**kwargs)
        self.file_path = os.path.join(path_to_folder_data_raw, my_file_name)
        print(f"Pfad zur CSV-Datei: {my_file_name} erstellt.")

class CleanedData(_RawDataPath):
    def __init__(self, 
                 column_names_types: dict[str, str], 
                 **kwargs
                 ) -> None:
        
        super().__init__(**kwargs)
        self.df_build_notes = ['Daten bereinigen \n-----------------']
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
        self.df_build_notes.append('\nDaten tranformieren\n-----------------')
        self.df_tformed = self.df_cleaned.copy()
        self.df_tformed['day'] = (self.df_cleaned['date']-sd).dt.days + 1
        self.df_tformed = self.create_demand(self.df_tformed)
        self.df_tformed = self.n_sick_adjusted(self.df_tformed)
        self.df_tformed = self.predict_past_demand(self.df_tformed)
        self.trend_reg = t = DataTrend(self.df_tformed)
        self.df_tformed['calls_reg_pred'] = t.calls_reg_pred
        note = "Erfolgreich: Spalte 'calls_reg_pred' erstellt"
        self.df_build_notes.append(note)
        self.df_tformed['calls_reg_act_diff'] = t.calls_reg_act_diff
        note = "Erfolgreich: Spalte 'calls_reg_act_diff' erstellt"
        self.df_build_notes.append(note)
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
        self.df_build_notes.append('\nFeatured erstellen\n-----------------')
        self.tup_features = features
        self.df_features = self.add_date_features(self.df_tformed,
                                                  self.tup_features)
        self.df_only_features = self.df_features[list(self.tup_features)]
        print('Features hinzugefügt')
    
    def add_date_features(self,
                             df: pd.DataFrame,
                             features: tuple[str],
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

    def vor_364(self) -> pd.DataFrame:
        df = self.df_features[['date', 'calls_reg_act_diff']].copy()
        df_1 = df.copy()
        df_1.rename(columns={'date':'date_vor_364', 'calls_reg_act_diff':'calls_rad_vor_364'}, inplace=True)
        df['date_vor_364'] = self.df_features['date'] - pd.Timedelta(days=364)
        df = pd.merge(df, df_1, on='date_vor_364', how='left')
        df.set_index('date', inplace=True)
        df.dropna(subset=['calls_rad_vor_364'], inplace=True)

        return df

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
        self.df_pred['adabooreg_reg_calls_pred'] = pred = self.final_ada_pred
        self.df_pred['adabooreg_error'] = pred - self.df_pred['calls']
        sby_need_pred = self.df_predict_sby_need(self.df_pred['date'])
        self.df_pred['adabooreg_sby_need'] = sby_need_pred['sby_need_pred']
        
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
                              startdate: 'datetime64',
                              df_pred: pd.DataFrame
                                ) -> np.array:

        # Regression für Notrufe
        reg_trend = DataTrend(df_pred).model_ftd
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
                                                              self.startdate,
                                                              self.df_pred)

        # Vorhersage des AdaBoost-Regressors
        df_date = pd.DataFrame(date, columns=['date'])
        df_pred = self.add_date_features(df_date, self.tup_features)
        df_pred_2 = df_pred[list(self.tup_features)]
        adabooreg_pred = self.model_adabooreg.adabooreg_ftd.predict(df_pred_2).reshape(-1)

        # gesamte Vorhersage von Notrufen
        pred = calls_reg_pred + adabooreg_pred

        # Vorhersage der Nachfrage an Einsatzfahrenden
        model_reg = RegressionCallsDemand(self.df_pred)
        model_reg_fitted = model_reg.model_ftd_reg_calls_demand
        sby_plus_duty = model_reg_fitted.predict(pred.reshape(-1, 1))

        # Subtrahieren von n_duty, vorausgestetzt n_duty ist 1900

        sby_need_1900 = np.round(((sby_plus_duty - 1900).reshape(-1)), 0).astype(int)

        s_sby_need = pd.Series(sby_need_1900, name='sby_need_pred')

        df = pd.concat([df_pred['date'], s_sby_need], axis=1)

        return df

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
        plt.savefig(f'{cd}\\abbildungen\\demand_vs_calls.jpg')

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
        to_save_fig = f'{cd}\\abbildungen\\overview_scatter.jpg'
        plt.savefig(to_save_fig)

    def plot_notruf_reg(self) -> None:
        # Streuungsdiagramm mit 'date' auf der x-Achse
        # und 'calls' als Punkte und 'calls_pred' als Linie
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.scatter(self.df['date'], self.df['calls'], s=3)
        ax.plot(self.df['date'], self.df['calls_reg_pred'], color='red')
        plt.savefig(f'{cd}\\abbildungen\\notruf_reg.jpg')

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
        to_save_fig = f'{cd}\\abbildungen\\no_trend_scatter.jpg'
        plt.savefig(to_save_fig)

    def scatter_compare(self, column_name_1, column_name_2, title) -> None:
        fig, ax = plt.subplots(figsize=(10, 5))

        ax.scatter(self.df['date'], 
                   self.df[column_name_1], 
                   s=3, 
                   color='lightgrey',
                   label=column_name_1)
        
        ax.scatter(self.df['date'], 
                   self.df[column_name_2],
                   s=3,
                   label=column_name_2)

        ax.legend()

        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Stationäre Nachfrage')

        plt.savefig(f'{cd}\\abbildungen\\demand_vs_adaboo.jpg')
    
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
        fig, (ax_1, ax_2, ax_3, ax_4, ax_5, ax_6) = plt.subplots(6, 1, sharex=True, figsize=(10, 16))
        test_days = 47
        date = self.df['date'][-test_days:]
        y_1 = self.df['calls'][-test_days:]
        y_2 = self.df['adabooreg_reg_calls_pred'][-test_days:]
        y_3 = self.df['adabooreg_error'][-test_days:]
        ada = self.df['adabooreg_sby_need'][-test_days:]
        y_4 = np.where(ada > 0, ada, 0)
        y_5 = self.df['sby_need'][-test_days:]
        y_6 = y_4 - y_5

        # Horizontalen Linien wo sby_needed > 0
        ax_1.axhline(y=9500, color='red', linestyle='--', label='sby_needed')
        ax_2.axhline(y=9500, color='red', linestyle='--')

        ax_1.legend()

        # Beide y-Achsen in einem figure als Streuungsdiagramme mit kleinen Punkten
        ax_1.scatter(date, y_1, color='blue', marker='.')
        ax_2.scatter(date, y_2, color='blue', marker='.')
        ax_3.scatter(date, y_3, color=np.where(y_3 >= 0, 'blue', 'red'), marker='.')
        ax_4.scatter(date, y_4, color='blue', marker='.')
        ax_5.scatter(date, y_5, color='red', marker='.')
        ax_6.scatter(date, y_6, color=np.where(y_6 >= 0, 'blue', 'red'), marker='.')

        # Achsenbeschriftung
        ax_6.set_xlabel('Datum')
        ax_1.set_ylabel('Calls')
        ax_2.set_ylabel('Predicted Calls')
        ax_3.set_ylabel('Predicted Calls - Calls')
        ax_4.set_ylabel('Predicted sby_need')
        ax_5.set_ylabel('Actual sby_need')
        ax_6.set_ylabel('Predicted - actual sby_need')
        
        plt.savefig(f'{cd}\\abbildungen\\plot_train_test.jpg')
        
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
        self.tss = tss = self._make_time_series_split()
        self.df_model_performance = self._model_cross_val(X, y, 
                                                         tup_models, scoring,
                                                         tss)
   
    def _make_time_series_split(self) -> TimeSeriesSplit:
        tss = TimeSeriesSplit(n_splits=7, test_size=32, gap=15)
        return tss
    
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
    def __init__(self,
                 tss,
                 dict_param_grid,
                 df_gscv_new_results,
                 df_gscv_all_results,
                 scoring
                 ) -> None:
        self.tss = tss
        self.dict_param_grid = dict_param_grid
        self.df_gscv_new_results = df_gscv_new_results
        self.df_gscv_all_results = df_gscv_all_results
        self.scoring = scoring

    @classmethod
    def run_adabooreg_gscv(cls, tss, param_grid):

        #kwargs = Config.dict_config
        cls_featured_data = cls_fd = FeaturedData(**(Config.dict_config))
        y = cls_fd.df_features['calls_reg_act_diff']
        X = cls_fd.df_only_features

        # adaboost regressor erstellen
        adabr = AdaBoostRegressor(estimator=DecisionTreeRegressor(), 
                                random_state=42)
        
        scoring = 'neg_mean_squared_error'
        adaboo_gscv = GridSearchCV(adabr, param_grid=param_grid, 
                        scoring=scoring, cv=tss)
        
        adaboo_gscv.fit(X, y)

        # sort dataframe by rank_test_score
        adaboo_gscv_df = pd.DataFrame(adaboo_gscv.cv_results_)
        adaboo_gscv_df = adaboo_gscv_df.sort_values(by='rank_test_score')

        df_to_append = pd.read_pickle(f'{cd}\\df_adaboo_gscv.pkl')
        appended = pd.concat([df_to_append, adaboo_gscv_df], axis=0, 
                             ignore_index=True, sort=False)

        appended.sort_values(by=['mean_test_score'], ascending=False, 
                             ignore_index=True)
        
        # speichere das DataFrame als pickle
        appended.to_pickle(f'{cd}\\df_adaboo_gscv.pkl')

        return cls(
            tss=tss,
            dict_param_grid=param_grid,
            df_gscv_new_results=adaboo_gscv_df,
            df_gscv_all_results=appended,
            scoring=scoring
        )