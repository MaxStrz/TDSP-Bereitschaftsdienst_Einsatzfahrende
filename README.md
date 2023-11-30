"Vorhersage für den Bereitschaftsdienstplan" 

## How to use.
Mache eine Vorhersage: [Vorhersagen.ipynb](https://github.com/MaxStrz/TDSP-Bereitschaftsdienst_Einsatzfahrende/blob/master/Code/Modeling/Vorhersagen.ipynb) - Das ist der wichtigste Notebook. Man kann Daten in Form "2016-01-29" eingeben und eine Vorhersage der Notwendigen Einsatzfahrende in Bereitschaftsdienst zurück kriegen.

### Analysis
Bereinige die Daten:  
Transformiere die Daten:  
Füge Features hinzu:  

### Modeling
Modell-Kreuzvalidierung:  
Grid-Search-Kreuzvalidierung:  
Train und Test eines AdaBoost-Modell:  
Speichere das Vorhersagemodell mit MLFlow:  


## Weitere Dateien / Module


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


# TDSP Project Structure, and Documents and Artifact Templates

This is a general project directory structure for Team Data Science Process developed by Microsoft. It also contains templates for various documents that are recommended as part of executing a data science project when using TDSP. 

[Team Data Science Process (TDSP)](https://docs.microsoft.com/en-us/azure/machine-learning/team-data-science-process/overview) is an agile, iterative, data science methodology to improve collaboration and team learning. It is supported through a lifecycle definition, standard project structure, artifact templates, and [tools](https://github.com/Azure/Azure-TDSP-Utilities) for productive data science. 


**NOTE:** In this directory structure, the **Sample_Data folder is NOT supposed to contain LARGE raw or processed data**. It is only supposed to contain **small and sample** data sets, which could be used to test the code.

The two documents under Docs/Project, namely the [Charter](./Docs/Project/Charter.md) and [Exit Report](./Docs/Project/Exit%20Report.md) are particularly important to consider. They help to define the project at the start of an engagement, and provide a final report to the customer or client.

**NOTE:** In some projects, e.g. short term proof of principle (PoC) or proof of value (PoV) engagements, it can be relatively time consuming to create and all the recommended documents and artifacts. In that case, at least the Charter and Exit Report should be created and delivered to the customer or client. As necessary, organizations may modify certain sections of the documents. But it is strongly recommended that the content of the documents be maintained, as they provide important information about the project and deliverables.
