import numpy as np
# [
# array(
# [[[692, 487]], [[684, 493]], [[680, 493]],  [[737, 489]], [[725, 489]]]
# , dtype=int32
# )
# ]
# Erstellen des Arrays
array = np.array([[[692, 487]], [[684, 493]], [[680, 493]], [[737, 489]], [[725, 489]]], dtype=np.int32)
# Bestimmen der Länge des Arrays
type = type(array)  # Anzahl der äußeren Elemente
print("Länge des Arrays:", type) #<class 'numpy.ndarray'>

# Bestimmen der Länge des Arrays
length = len(array)  # Anzahl der äußeren Elemente
print("Länge des Arrays:", length) #Länge des Arrays: 5

# Alternativ kannst du auch die Form des Arrays verwenden
shape = array.shape
print("Form des Arrays:", shape) #Form des Arrays: (5, 1, 2)
