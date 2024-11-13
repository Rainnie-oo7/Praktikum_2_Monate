from scipy.io import loadmat
import numpy as np

# Lade die .mat-Datei
mat_file_path = '/home/boris.grillborzer/PycharmProjects2/00CNNundRNN/boris.grillborzer/11HandDatasetFirst/polygons.mat'
mat_data = loadmat(mat_file_path)

# Text sammeln
output_text = ""

# Schlüssel der Datei anzeigen und in Text speichern
output_text += "Schlüssel in der .mat-Datei:\n"
output_text += str(mat_data.keys()) + "\n\n"

# Wähle den relevanten Schlüssel der Datenstruktur
data_key = 'polygons'  # Ersetze 'PennFudanPed' durch den tatsächlichen Schlüssel in der Datei
data = mat_data[data_key]

# Anzeige des "Head" der Daten (z.B. die ersten 5 Zeilen)
output_text += "Head der Daten:\n"
output_text += str(data[:5]) + "\n\n" if isinstance(data, np.ndarray) else "Datenstruktur ist kein Array\n\n"

# Anzeige des "Tail" der Daten (z.B. die letzten 5 Zeilen)
output_text += "Tail der Daten:\n"
output_text += str(data[-5:]) + "\n\n" if isinstance(data, np.ndarray) else "Datenstruktur ist kein Array\n\n"

# Speichere die Ausgabe als Textdatei
with open("mat_file_output.txt", "w") as file:
    file.write(output_text)

print("Ausgabe erfolgreich als 'mat_file_output.txt' gespeichert.")
