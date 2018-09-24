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