import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def csv_to_dataframe():

	filepath = "../Sample_Data/Raw/sickness_table.csv"
	dtypen = {
		'n_sick':'int64',
		'calls':'int64',
		'n_duty':'int64',
		'n_sby':'int64',
		'sby_need':'int64',
		'dafted':'int64'}
	# Daten aus der CSV-Datei in einen Pandas DataFrame
	df = pd.read_csv(filepath, index_col=0, dtype=dtypen, parse_dates=['date'])
	return df

def calls_scatter(df):

	# Erstelle das Streuungsdiagramm erneut mit modifizierten Beschriftungen für die x-Achse
	fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(10, 10))
	date = df['date']
	calls = df['calls']
	n_sick = df['n_sick']
	sby_need = df['sby_need']
	dafted = df['dafted']

	ax1.scatter(date, calls, s=3)
	ax2.scatter(date, n_sick, s=3)
	ax3.scatter(date, sby_need, s=3)

	# Hauptmarkierung der x-Achse Monatsnamen in abgekürzter Form
	ax3.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
	# Hauptmarkierung der x-Achse mit Interval von 3 Monaten
	ax3.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
	# Nebenmarkierung der x-Achse als Jahr
	ax3.xaxis.set_minor_formatter(mdates.DateFormatter('%Y'))
	# Nebenmarkierung für jedes Jahr
	ax3.xaxis.set_minor_locator(mdates.YearLocator())

	# Abstand der Jahr-Beschriftungen vom Plot vergrößen
	ax3.tick_params(axis='x', which='minor', pad=25)

	# Beschriftung der Achsen und Titel
	ax1.set_ylabel('Anzahl der Notrufe')
	ax1.set_title('Streuungsdiagramm: Notrufe und Krankmeldungen nach Datum')
	ax2.set_ylabel('Krank gemeldet')
	ax3.set_ylabel('Bereitschaftsdienst aktiviert')
	ax3.set_xlabel('Datum')


	# Zeige das Diagramm
	plt.show()

def describe_data(df):
	n_sby = df['n_sby'].unique()
	n_duty = df['n_duty'].unique()
	max_notrufe = max(df['calls'])
	print(f"Werte in der n_sby-Spalte:{n_sby}")
	print(f"Werte in der n_duty-Spalte:{n_duty}")