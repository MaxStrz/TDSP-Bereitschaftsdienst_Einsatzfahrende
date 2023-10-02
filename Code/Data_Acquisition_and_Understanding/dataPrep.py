import pandas as pd
import numpy as np  
import os
from pandas.tseries.offsets import DateOffset
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import featuretools as ft
from featuretools.primitives import Lag, RollingMin, RollingMean
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import warnings


# ignoriere FutureWarnings
warnings.simplefilter(action='ignore', category=(FutureWarning, pd.errors.PerformanceWarning))

def my_df():
    """Liest die CSV-Datei ein, überprüft die Datenqualität und gibt einen DataFrame zurück"""

    df_build_notes = []
    def csv_filepath2df():
        """Verwendet absoluten Pfad des Skripts und relativen Pfad zur CSV-Datei um einen DataFrame zu erstellen"""
        current_directory = os.path.dirname(__file__) # <-- absolute dir the script is in
        filepath = os.path.join(current_directory, "../../Sample_Data/Raw/sickness_table.csv") # <-- absolute dir the script is in + path to csv file
        df = pd.read_csv(filepath, index_col=0)

        # Bestätigung, dass die CSV-Datei erfolgreich in einen DataFrame umgewandelt wurde
        df_build_notes.append("Daten erfolgreich in einen DataFrame umgewandelt")

        return df
    
    df = csv_filepath2df()

    def missing_data(df):
        # Leere Liste für fehlende Daten
        missing_data_list = []

        for col in df.columns:
            # Überprüft ob es fehlende Daten in den jeweiligen Spalten gibt.
            missing_data = df[col].isna().sum()
            if missing_data != 0:
                # for-Schleife um die fehlenden Daten in der jeweiligen Spalte zu finden
                for index, value in enumerate(df[col]):
                    if pd.isna(value):
                        # Füge die Spalte, das Datum und den Index des fehlenden Wertes in die Liste ein
                        missing_data_list.append([col, df['date'][index], index])
            else:
                # nächste Spalte
                continue

        # Wenn die Liste nicht leer ist, raise error
        if len(missing_data_list) != 0:
            # Beschreibe die fehlenden Daten
            print(f"Es fehlen {len(missing_data_list)} Daten im Datensatz:")
            for data in missing_data_list:
                print(f"Spalte: {data[0]}, Datum: {data[1]}, Index: {data[2]}")
            raise ValueError("Fehlende Daten in der CSV-Datei")
        else:
            df_build_notes.append("Keine fehlenden Daten in der CSV-Datei")
    
    missing_data(df)
    
    def sind_sie_ganzzahlig(df):
        # Leere Liste für nicht-ganzzahlige Werte
        non_int_list = []
        # Überprüft ob alle Werte in der Spalte 'calls', 'sby_need', 'dafted', 'n_sick', 'n_duty', 'n_sby' gleich ihrem Interger-Wert sind.
        # Wenn nicht, raise error und gebe das Datum aus der 'date'-Spalte und Index des fehlerhaften Wertes aus.
        for col in ['calls', 'sby_need', 'dafted', 'n_sick', 'n_duty', 'n_sby']:
            for index, value in enumerate(df[col]):
                if value != int(value):
                    # Füge die Spalte, das Datum und den Index des fehlenden Wertes in die Liste ein
                    non_int_list.append([col, df['date'][index], index])
                else:
                    continue
        
        # Wenn die Liste nicht leer ist, beschreibe die fehlerhaften Daten und raise error
        if len(non_int_list) != 0:
            print(f"Es gibt {len(non_int_list)} nicht-ganzzahlige Werte im Datensatz:")
            for data in non_int_list:
                print(f"Spalte: {data[0]}, Datum: {data[1]}, Index: {data[2]}")
            raise ValueError("Nicht-ganzzahlige Werte in den Spalten 'calls', 'sby_need', 'dafted', 'n_sick', 'n_duty', 'n_sby'")
        else:
            df_build_notes.append("Keine nicht-ganzzahligen Werte in den Spalten 'n_sick', 'calls', 'n_duty', 'n_sby', 'sby_need', 'dafted'")

    sind_sie_ganzzahlig(df)

    def daten_typen_umwandeln(df):
        # Alle Spalten außer 'date' in Integer umwandeln
        for col in ['calls', 'sby_need', 'dafted', 'n_sick', 'n_duty', 'n_sby']:
            df[col] = df[col].astype(int)

        # Bestätigung, dass alle Spalten außer 'date' in Integer umgewandelt wurden
        df_build_notes.append("Alle Spalten ausser 'date' in Integer umgewandelt")

        # 'date'-Spalte in Datetime umwandeln
        df['date'] = pd.to_datetime(df['date'])

        # Bestätigung, dass alle Daten in der 'date'-Spalte Datetime sind
        df_build_notes.append("Alle Daten in der 'date'-Spalte sind Datetime-Datentyp")

        return df
    
    df = daten_typen_umwandeln(df)

    def alle_Daten_vorhanden(df):
        """Überprüft, ob alle Daten zwischen Start- und Enddatum vorhanden sind"""
        start_date = df['date'].min()
        end_date = df['date'].max()
        date_range = pd.date_range(start=start_date, end=end_date)
        missing_dates = []
        for date in date_range:
            if date not in df['date'].values:
                missing_dates.append(date)
        if len(missing_dates) == 0:
            df_build_notes.append("Alle Daten zwischen Start- und Enddatum vorhanden")
        else:
            print(f"Es fehlen {len(missing_dates)} Daten zwischen Start- und Enddatum")
            print(f"Die fehlenden Daten sind: {missing_dates}")
        
        return df
    
    df = alle_Daten_vorhanden(df)

    def df_summary(df):
        """Gibt eine Zusammenfassung des DataFrames aus"""
        summary_list = []
        summary_list.append("\nDataframe Info:")
        summary_list.append(df.describe())
        summary_list.append(df.head())
        summary_list.append(df.tail())

        return summary_list
     
    summary_list = df_summary(df)

    return df, df_build_notes, summary_list

def df_new_columns(df):
    df['month'] = df['date'].dt.month # Monat als Zahl (1-12)
    #df['year'] = df['date'].dt.year # Jahr (4-stellig)
    #df['dayofmonth'] = df['date'].dt.day # Tag des Monats (1-31)
    #df['weekday'] = df['date'].dt.weekday # Wochentag als Zahl (Montag = 0, Sonntag = 6)
    #df['weekofyear'] = df['date'].dt.isocalendar().week # Kalenderwoche als Zahl (1-52)
    #df['dayofyear'] = df['date'].dt.dayofyear # Tag des Jahres als Zahl (1-365)
    df['season'] = (df['month']-1) % 12 // 3 + 1 # Jahreszeit als Zahl (1-4) (1 = Winter, 2 = Frühling, 3 = Sommer, 4 = Herbst)
    df['predict_day'] = df['date'] - DateOffset(months=1, day=15) # Datum des 15. des vorherigen Monats

    # Nachfrage = aktivierter Bdienst + \regulären Dienst - krank gemeldet (wenn 'sby_need' > 0)
    df['demand'] = np.where(df['sby_need'] > 0, df['sby_need'] + df['n_duty'] - df['n_sick'], np.nan) 
    
    return df

# def sby_needed(df):
#     # Entferne alle Zeilen, in denen 'sby_need' = 0 ist
#     df = df.query('sby_need != 0')

#     return df

def sby_needed_vs_calls(df):
    # Erstelle ein Streuungsdiagramm mit 'sby_need' auf der x-Achse und 'calls' auf der y-Achse
    fig, ax = plt.subplots()
    ax.scatter(df['calls'], df['demand'], s=3)
    # Beschriftung der Achsen und Titel
    ax.set_title('Streuungsdiagramm: Notrufe und Gesamtnachfrage')
    ax.set_xlabel('Anzahl der Notrufe')
    ax.set_ylabel('Gesamtnachfrage')
    # Zeige das Diagramm
    plt.show()

def overview_scatter(df):

    # Erstelle das Streuungsdiagramm erneut mit modifizierten Beschriftungen für die x-Achse
    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, sharex=True, figsize=(10, 10))
    date = df['date']
    calls = df['calls']
    n_sick = df['n_sick']
    sby_need = df['sby_need']
    n_duty = df['n_duty']
    dafted = df['dafted']

    ax1.scatter(date, calls, s=3)
    ax2.scatter(date, n_sick, s=3)
    ax3.scatter(date, sby_need, s=3)
    ax4.scatter(date, dafted, s=3)
    ax5.scatter(date, n_duty, s=3)

    # Hauptmarkierung der x-Achse Monatsnamen in abgekürzter Form
    ax5.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    # Hauptmarkierung der x-Achse mit Interval von 3 Monaten
    ax5.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    # Nebenmarkierung der x-Achse als Jahr
    ax5.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
    # Nebenmarkierung für jedes Jahr
    ax5.xaxis.set_minor_locator(mdates.YearLocator())

    # Abstand der Jahr-Beschriftungen vom Plot vergrößen
    # ax1.set_xticks([])
    # ax2.set_xticks([])
    # ax3.set_xticks([])
    # ax4.set_xticks([])
    for ax in range(4):
        fig.axes[ax].tick_params(axis='x', which='both', length=0)
    ax5.tick_params(axis='x', which='minor', length=10, pad=25)
    ax5.tick_params(axis='x', which='major', length=5)

    # Beschriftung der Achsen und Titel
    ax1.set_title('Streuungsdiagramm: Notrufe und Krankmeldungen nach Datum')
    ax1.set_ylabel('Anzahl der Notrufe')
    ax2.set_ylabel('Krank gemeldet')
    ax3.set_ylabel('Bereitschaftsdienst \naktiviert')
    ax4.set_ylabel('Zusätzliche \nEinsatzfahrende \naktiviert')
    ax5.set_ylabel('Einsatzfahrende \nim Dienst')
    ax5.set_xlabel('Datum')

    for ax in fig.get_axes():
        ax.set_ylim(bottom=0) # y-Achse für alle Axen in Figure mit 0 beginnen lassen
        ax.set_ylim(top=ax.get_ylim()[1]*1.05) # y-Achse für alle Axen in Figure + 5% des höchsten Wert der jeweiligen Spalte

    # Zeige das Diagramm
    plt.show()

def describe_data(df):
    n_sby = df['n_sby'].unique()
    n_duty = df['n_duty'].unique()
    max_notrufe = max(df['calls'])
    print(f"Werte in der n_sby-Spalte:{n_sby}")
    print(f"Werte in der n_duty-Spalte:{n_duty}")

def features(df):

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

    data = train_test_daten(df) # Datensatz in Trainings- und Testdaten mit Spalte 'test_data' aufteilen

    es = ft.EntitySet('einsatzfahrende') # Erstelle EntitySet mit Namen 'einsatzfahrende'

    es.add_dataframe( # Füge Dataframe 'data' zum EntitySet hinzu
        data,
        dataframe_name='data',
        index='index',
        time_index='date'
        )
    
    # Lücke zwischen Zielzeitpunkt und letztem Zeitpunkt im Datensatz für den Feature-Berechnungszeitpunkt
    gap = 16
    # Zeitfenster für den Feature-Engineering
    window_length = 7

    # Delaying-Primitives  
    delaying_primitives = [ft.primitives.Lag(periods=i + gap) for i in range(window_length)]

    rolling_mean_primitive = RollingMean(window_length=window_length, 
                                         gap=gap, min_periods=window_length)

    rolling_min_primitive = RollingMin(window_length=window_length, 
                                       gap=gap, min_periods=window_length)
    
    # Erstelle Features mit Deep Feature Synthesis
    # 'day' ist Tag des Monats (1-31)
    # 'week' ist Kalenderwoche (1-52)
    datetime_primitives = ['month', 'weekday', 'is_weekend', 'week', 'day', 'year', 'day_of_year']

    features, feature_names = ft.dfs(entityset=es, # EntitySet
                                    target_dataframe_name='data', # Ziel-Dataframe	
                                    trans_primitives=(
                                        datetime_primitives
                                        + delaying_primitives
                                        + [rolling_mean_primitive, rolling_min_primitive]),
                                    cutoff_time=data['date'].min(), # Startdatum für Feature-Berechnung
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

def notrufe_demand_reg(df):

    # Print df-Spalten 'calls' and 'demand' where 'demand' is not NaN.
    df_reg = df[['calls', 'demand']].query('demand > 0')
    x = np.array(df_reg['calls'])
    x = x.reshape(-1, 1) # x muss 2D sein weil LinearRegression.fit() 2D-Array erwartet
    y = np.array(df_reg['demand']).reshape(-1, 1)

    reg = LinearRegression().fit(x, y)
    reg_score = reg.score(x, y)

    demand_pred = reg.predict(np.array(df['calls']).reshape(-1, 1))
    demand_pred = (np.round(demand_pred, 0)).astype(int)
    df['demand_pred'] = demand_pred

    return reg, reg_score, df


