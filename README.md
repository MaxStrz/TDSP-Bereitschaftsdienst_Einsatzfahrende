## Vorhersage für den Bereitschaftsdienstplan  

## How to use.
Mache eine Vorhersage: [Vorhersagen.ipynb](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/Vorhersagen.ipynb) - Das ist der wichtigste Notebook. Man kann Daten in Form "2016-01-29" eingeben und eine Vorhersage der Notwendigen Einsatzfahrende in Bereitschaftsdienst zurück kriegen.  

### Analysis
[Bereinige die Daten](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Analysis/Datenbereinigung.ipynb)  
[Transformiere die Daten](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Analysis/Datentransformation.ipynb)  
[Füge Features hinzu](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Analysis/Features.ipynb)  
[Abbildungen](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/tree/master/Code/Analysis/abbildungen)  
  
### Modeling  
[Benchmarkmodell](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/Benchmarkmodell.ipynb)  
[Modell-Kreuzvalidierung](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/Modelle_Kreuzvalidierung.ipynb)  
[Grid-Search-Kreuzvalidierung](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/AdaBooReg_GridSearchKreuzvalidierung.ipynb)  
[Train und Test eines AdaBoost-Modell](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/AdaBooReg_ModellTrainTest.ipynb)  
[Speichere das Vorhersagemodell mit MLFlow](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/Vorhersagemodell_speichern.ipynb)  

## TODO
- Erklärebarkeit (Explainability) muss gemacht werden.
- Schreibe Unit-Tests
- Vorhersagemodell sollte nicht zweimal geschrieben werden, einmal in MLFlow und einmal in dataPrep.
- Vorhersagemodell sollte nicht mittels eines runs gelogged werden.
- DataPrediction-Klasse und df_predict_sby_need-Methode muss aktualisiert werden, um es zu vereinfachen, das fertige Modell für Production zu verpacken.
- dataPrep neu benennen. Vielleicht 'project_code'.
- Config als json-Datei oder ähnlich statt Klasse in dataPrep.
- 'Single source of truth' für Anzahl an Testtagen in dataPrep.
- Konstruktoren sind oft zu groß. Verschiebe Logik zu Builders / Factorys.
- dataPrep muss mit Docstrings und Kommentare besser dokumentiert sein.
- Notebooks sollte mehr Aufbau/Prozesslogik übernehmen. Im Notebook Bereitschaftsdienstplan_Datentransformation beispielsweise erzeugt der Notebook bloß eine Klasseninstanz und ruft danach Attribute an. Die Transformationen selbst werden in dem Konstruktor ausgeführt, was möglicherweise nicht ideal ist, obwohl es macht es ganz klar, in welchem Zustand sich der DataFrame befindet.
- Manche Methode der Klassen sollte private Methoden sein d.h. mit _ anfangen.
- In der Bereinigungs, Transformations und Feature-Phasen, setzte DVC um.
- status-Spalte wird nicht verwendet.
- Weitere Versuche mit ARIMA und SARIMA-Modelle. Mindesten um die wichtigkeit von vorherigen Datenpunkten zu bestimmen.
- Features zu probieren: Wert vom Jahr davor, Wert von 2., 3. usw. vorherigen Monat.

## DOING
