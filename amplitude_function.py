import numpy as np
import matplotlib.pyplot as plt

def amplitude(xv: float, yv: float, n: int, m: int) -> float: 
    return np.abs(np.sin(n*np.pi*xv/2)*np.sin(m*np.pi*yv/2)-np.sin(m*np.pi*xv/2)*np.sin(n*np.pi*yv/2))

# Crear una malla de puntos en el rango [0, 100] x [0, 100]
x = np.linspace(-1, 1, 100)
y = np.linspace(-1, 1, 100)
X, Y = np.meshgrid(x, y)

# Calcular los valores de la función amplitude para cada punto
Z = amplitude(X, Y, 3, 3.5)

# Visualizar usando matshow
plt.matshow(Z, cmap='viridis', origin='lower', extent=[0, 1, 0, 1])
plt.colorbar(label='Amplitude')
plt.title('Función Amplitude')
plt.xlabel('xv')
plt.ylabel('yv')
plt.show()