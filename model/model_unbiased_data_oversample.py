# Data
import numpy as np
import pandas as pd
from sklearn import preprocessing
# Machine Learning algorithms
import sklearn
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
# Metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
# Neural Network
from keras.models import Sequential
from keras.optimizers import SGD, Adam, adam
from keras.layers import Dense, Dropout


## Data Preprocessing ##

train = pd.read_csv("..\\data\\verkehrsunfaelle_train.csv", header = 0,encoding='latin-1')
test = pd.read_csv("..\\data\\verkehrsunfaelle_test.csv", header = 0,encoding='latin-1')

# Entferne erste Index Spalte
train = train.iloc[:,1:]
test = test.iloc[:,1:]

# Feature Scaler für Normalisierung der numerischen Features
scaler = preprocessing.MinMaxScaler(feature_range=(0, 1))

# Behandlung der einzelnen Features

# Unfalldatum
def date_code(date):

    if date.lower().__contains__("jan"):
        return 0
    elif date.lower().__contains__("feb"):
        return 1
    elif date.lower().__contains__("mar") or date.lower().__contains__("mrz") :
        return 2
    elif date.lower().__contains__("apr"):
        return 3
    elif date.lower().__contains__("may") or date.lower().__contains__("mai") :
        return 4
    elif date.lower().__contains__("jun"):
        return 5
    elif date.lower().__contains__("jul"):
        return 6
    elif date.lower().__contains__("aug"):
        return 7
    elif date.lower().__contains__("sep"):
        return 8
    elif date.lower().__contains__("okt") or date.lower().__contains__("oct") :
        return 9
    elif date.lower().__contains__("nov"):
        return 10
    elif date.lower().__contains__("dez") or date.lower().__contains__("dec") :
        return 11

train["Unfalldatum"] = train["Unfalldatum"].apply(date_code)
test["Unfalldatum"] = test["Unfalldatum"].apply(date_code)

train["Unfalldatum"] = scaler.fit_transform(train[["Unfalldatum"]])
test["Unfalldatum"] = scaler.transform(test[["Unfalldatum"]])

# Alter
train["Alter"] = np.floor(train["Alter"]/10)
test["Alter"] = np.floor(test["Alter"]/10)

train["Alter"] = scaler.fit_transform(train[["Alter"]])
test["Alter"] = scaler.transform(test[["Alter"]])

# Verletzte Personen
#print(train.groupby(["Verletzte Personen","Unfallschwere"]).size())
train["Verletzte Personen"] = scaler.fit_transform(train[["Verletzte Personen"]])
test["Verletzte Personen"] = scaler.transform(test[["Verletzte Personen"]])

# Anzahl Fahrzeuge
train["Anzahl Fahrzeuge"] = scaler.fit_transform(train[["Anzahl Fahrzeuge"]])
test["Anzahl Fahrzeuge"] = scaler.transform(test[["Anzahl Fahrzeuge"]])

#  Datenpunkt (Wert:9) Korrektur
train.at[9380,'Bodenbeschaffenheit'] = "trocken"

# Zeit (24h)
train["Zeit (24h)"] = np.floor(train["Zeit (24h)"]/100)
test["Zeit (24h)"] = np.floor(test["Zeit (24h)"]/100)

def code_time(time):
    if time >=4.0 and time < 10.0:       #morgen
        return 0
    elif time >= 10.0 and time < 18.0:   #tag
        return 1
    else:                                #nacht
        return 2

train["Zeit (24h)"] = train["Zeit (24h)"].apply(code_time)
test["Zeit (24h)"] = test["Zeit (24h)"].apply(code_time)

train["Zeit (24h)"] = scaler.fit_transform(train[["Zeit (24h)"]])
test["Zeit (24h)"] = scaler.transform(test[["Zeit (24h)"]])

# One Hot encoding der kategorischen Features
train_oh = pd.get_dummies(data=train, columns=['Strassenklasse', 'Unfallklasse', 'Lichtverhältnisse', 'Bodenbeschaffenheit', 'Geschlecht'
                                               , 'Fahrzeugtyp', 'Wetterlage'])
test_oh = pd.get_dummies(data=test, columns=['Strassenklasse', 'Unfallklasse', 'Lichtverhältnisse', 'Bodenbeschaffenheit', 'Geschlecht'
                                               , 'Fahrzeugtyp', 'Wetterlage'])

# Konvertierung zu float32 Datentyp für besseres GPU Training
train_oh = train_oh.astype(np.float32)
test_oh = test_oh.astype(np.float32)

### Auswahl der Features mit höchster Correlation zu Unfallschwere
#
# Features die dabei entfernt werden:
# {'Lichtverhältnisse_Dunkelheit: Strassenbeleuchtung unbekannt', 'Strassenklasse_nicht klassifiziert',
# 'Bodenbeschaffenheit_Frost/ Ice', 'Wetterlage_Gut (starker Wind)', 'Bodenbeschaffenheit_Frost / Eis',
# 'Bodenbeschaffenheit_Überflutung', 'Wetterlage_Unbekannt', 'Fahrzeugtyp_Kleinbus',
# 'Fahrzeugtyp_Transporter', 'Bodenbeschaffenheit_nass / feucht', 'Fahrzeugtyp_LKW bis 7.5t',
# 'Strassenklasse_Bundesstrasse', 'Wetterlage_Nebel', 'Strassenklasse_unbefestigte Strasse',
# 'Unfalldatum', 'Strassenklasse_Landesstrasse', 'Bodenbeschaffenheit_trocken'}

featureselection = True
if featureselection:
    corr = train_oh.corr()
    corr_to_target = corr["Unfallschwere"]
    treshholded_columns_bool = abs(corr_to_target) >= 0.01 # niedriger threshold da correlation insgesamt niedrig ausfällt
    selected_columns = corr_to_target[treshholded_columns_bool].index.tolist()

    train_oh = train_oh[selected_columns]
    # Entfernung der Features im Test Datensatz
    missing_cols_2 = set( test_oh.columns ) - set( train_oh.columns )
    for c in missing_cols_2:
        if c in test_oh.columns:
            test_oh = test_oh.drop(c,axis=1)
else:
    # Füge leere Spalte im Test set hinzu damit Spaltenanzahl übereinstimmt mit Trainingsdaten
    missing_cols = set( train_oh.columns ) - set( test_oh.columns )
    for c in missing_cols:
        test_oh[c] = 0

# Übereinstimmung der Spalten herstellen
train_oh,test_oh = train_oh.align(test_oh,join="left",axis=1)

### Daten Analyse:
# Übersicht über die Kategorien
#
#    Anzahl Unfallschwere Prozent der Gesamtdaten
#     13495 Kategorie 1   88.6 %
#      1618 Kategorie 2   10.6 %
#       108 Kategorie 3   00.8 %

### Oversampling
# Versuch die ungleiche Verteilung der Unfallschwere Kategorien durch Oversampling zu vermindern
first_cat = train_oh[train_oh["Unfallschwere"] == 1]
second_cat = train_oh[train_oh["Unfallschwere"] == 2]
third_cat = train_oh[train_oh["Unfallschwere"] == 3]

validation = False
if validation:
    # Validierungs Set
    first_test_sample_rows = np.random.choice(first_cat.index.values, 70, replace=False)
    first_test_sample = first_cat.ix[first_test_sample_rows]
    first_cat = first_cat.drop(first_test_sample.index.values)

    second_test_sample_rows = np.random.choice(second_cat.index.values, 20, replace=False)
    second_test_sample = second_cat.ix[second_test_sample_rows]
    second_cat = second_cat.drop(second_test_sample.index.values)

    third_test_sample_rows = np.random.choice(third_cat.index.values, 10, replace=False)
    third_test_sample = third_cat.ix[third_test_sample_rows]
    third_cat = third_cat.drop(third_test_sample.index.values)

### Oversampling von Kategorie 2 und 3

second_cat = second_cat.sample(10000, replace=True)
third_cat = third_cat.sample(10000, replace=True)

# Alles wieder zusammen fügen
train_data = pd.concat([first_cat,second_cat,third_cat],axis=0)
if validation:
    test_val_data = pd.concat([first_test_sample,second_test_sample,third_test_sample],axis=0)


# Unfallschwere entfernen und als Label One Hot enkodieren
train_labels = train_data["Unfallschwere"]
train_data.drop(["Unfallschwere"],axis=1, inplace=True)
# One Hot Encoding
train_labels_oh = pd.get_dummies(train_labels,columns=["Unfallschwere"])

if validation:
    # One Hot Encoding vom Validierungs Set
    test_val_labels = test_val_data["Unfallschwere"]
    test_val_data.drop(["Unfallschwere"],axis=1, inplace=True)
    test_labels_oh = pd.get_dummies(test_val_labels,columns=["Unfallschwere"])


    ### Model Training ###

    # Random Forest und Decision Tree Classifier als Vergleich für Neuronales Netz
    forest = RandomForestClassifier(n_estimators=100)
    multi_target_forest = MultiOutputClassifier(forest)
    multi_target_forest.fit(train_data, train_labels_oh)
    Y_pred = multi_target_forest.predict(test_val_data)

    # Metrics
    print(np.round(accuracy_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1)),2),"accuracy")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average=None),2),"f1 score")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average="weighted"),2),"f1 score weighted")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average="macro"),2),"f1 score macro")
    print(confusion_matrix(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1)))


    decision_tree = DecisionTreeClassifier()
    decision_tree.fit(train_data, train_labels_oh.values)
    Y_pred = decision_tree.predict(test_val_data)

    # Metrics
    print(np.round(accuracy_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1)),2),"accuracy")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average=None),2),"f1 score")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average="weighted"),2),"f1 score weighted")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1),average="macro"),2),"f1 score macro")
    print(confusion_matrix(test_labels_oh.values.argmax(axis=1), Y_pred.argmax(axis=1)))


# NN architecture
print("Training Neural Network")
clf = Sequential()
clf.add(Dense(1600,input_dim=train_data.shape[1], activation='relu'))
clf.add(Dense(1200,  activation='relu'))
clf.add(Dense(1000,  activation='relu'))
clf.add(Dense(3, activation='softmax'))

clf.compile(optimizer="adam", loss='categorical_crossentropy',metrics=['categorical_accuracy'])
clf.fit(train_data, train_labels_oh, batch_size=128, epochs=30, verbose=0,shuffle=1)

if validation:
    preds = clf.predict(test_val_data)
    classes = clf.predict_classes(test_val_data)

    # Metrics
    print(np.round(accuracy_score(test_labels_oh.values.argmax(axis=1), classes),2),"accuracy")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), classes,average=None),2),"f1 score")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), classes,average="weighted"),2),"f1 score weighted")
    print(np.round(f1_score(test_labels_oh.values.argmax(axis=1), classes,average="macro"),2),"f1 score macro")
    print(confusion_matrix(test_labels_oh.values.argmax(axis=1), classes),"confusion matrix")


# Test Daten evaluieren und als .csv speichern
print("PREDICTING TEST DATA")
test_oh.drop(["Unfallschwere"],axis=1, inplace=True)
test_predictions = clf.predict_classes(test_oh)
submission = pd.DataFrame(data=test_predictions,columns=["Unfallschwere"])
print(submission["Unfallschwere"].value_counts())
submission.to_csv("predictions.csv")



# Die Labels(Kategorien/Klassen der Unfallschwere) im Datensatz sind ungleich verteilt, Kategorie 1 ist mit 90% am häufigsten vertreten.
# Gefahr der Accuracy Falle:
# Mit einem Klassifizierer der nur Kategorie eins ausgibt erhält man 90% Genauigkeit
# und fehl-klassifiziert die anderen zwei Kategorien komplett.
# Statt Accuracy muss also ein anderes Fehlermaß benutzt werden.
#
# Ich schaute mir die Confusion Matrix an, die anzeigt wie die einzelnen Samples (fehl) klassifiziert wurden.
# Gegen den starken Kategorie eins Bias habe ich versucht anzukämpfen,
# indem ich die Datenpunkte der Kategorie zwei und drei auf die Anzahl der Kategorie eins erhöht habe.
# Dieser Vorgang heißt "oversampling" der unterrepresentierten Labels.
# Desweiteren habe ich versucht mit dem loss des f1 scores, die architektur meines Neuronalen Netzwerkes zu optimieren.
# Das Fully Connected Neuronale Netz besteht aus 3 hidden layern mit 1600/1200/1000 features und relu Aktivierungsfunktion
# Zum Vergleich nutzte ich den RandomForestClassifier und DecisionTreeClassifier und baute ein ValidationSet.
# Ich nutzte den f1-score mit macro average, der bei multiclass classification besser über die performance eines models informiert.
# Meine Correlation Analysis ergab außerdem, das alle Features nur schwach mit Unfallschwere korrelieren. (<= 0.2)
#
# Ich bin gespannt über das Endergebnis meines Classifiers