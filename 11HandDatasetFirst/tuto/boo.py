import matplotlib.pyplot as plt
import numpy as np

# Koordinaten des Polygons
polygon = np.array([[1, 2], [2, 3], [3, 1], [2, 1]])

# Füllen und Plotten
plt.fill(polygon[:, 0], polygon[:, 1], color='blue')
plt.xlim(0, 4)
plt.ylim(0, 4)
plt.axis('off')  # Achsen ausblenden
plt.savefig('polygon.png', bbox_inches='tight')
plt.show()
