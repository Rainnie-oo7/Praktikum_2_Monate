import numpy as np

# Beispiel-Daten für labels und maske
labels = [1, 2, 3, 4]
maske = [10, 20, 30, 40]
print(labels)
print(maske)
# Sammle die gepaarten Daten in einer Liste
data = [(i, l, m) for i, (l, m) in enumerate(zip(labels, maske))]

# Konvertiere die Liste in ein numpy-Array
array = np.array(data)
print(array)
# Reshape das Array in die gewünschte Form, z.B. eine 90 Grad Drehung (transpose)
rotated_array = np.transpose(array)

print(rotated_array)
