import matplotlib.pyplot as plt
import numpy as np

# Datos ajustados
cm = np.array([[22, 3],
               [8, 177]])

# --- Guardar matriz como figura ---
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest')
plt.title("Confusion matrix")
plt.colorbar()
plt.xticks([0, 1], ["ok", "defective"])
plt.yticks([0, 1], ["ok", "defective"])

# Texto dentro de las celdas
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center", color="white")

plt.savefig("confusion_matrix_adjusted.png")
plt.show()