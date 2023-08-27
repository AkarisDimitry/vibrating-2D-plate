# Importing necessary libraries and modules
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For plotting and visualization
from matplotlib.animation import FuncAnimation  # For creating animations
import time  # For timing operations
import logging  # For logging messages
from typing import List, Tuple  # For type hints
from scipy.spatial import cKDTree  # For spatial data structures and algorithms
import imageio  # For reading and writing image data
import unittest 
import argparse
import tkinter as tk
from tkinter import ttk
import pygame

# Setting up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A decorator to measure and log the execution time of functions
def cronometrar(func):
    def wrapper(*args, **kwargs):
        inicio = time.time()
        resultado = func(*args, **kwargs)
        fin = time.time()
        logging.debug(f"Function {func.__name__} took {fin - inicio:.4f} seconds to execute.")
        return resultado
    return wrapper

# Main class for the simulation
class Simulacion:
    def __init__(self, num_particulas: int, ancho: float, alto: float, radio_particula: float = 0.02) -> None:
        """ 
        Initializes the simulation. 
        
        Parameters:
        - num_particulas (int): Number of particles in the simulation.
        - ancho (float): Width of the simulation area.
        - alto (float): Height of the simulation area.
        - radio_particula (float, optional): Radius of each particle. Default is 0.02.
        
        Attributes:
        - posiciones (np.ndarray): 2D array containing the x and y positions of each particle.
        - velocidades (np.ndarray): 2D array containing the x and y velocities of each particle.
        - radio_particula (float): Radius of each particle.
        - N (int): Interval for updating particle velocities.
        - contador (int): Counter for the number of steps taken.
        - frames (list): List to store the paths of saved frames for gif or mp4 generation.
        - dt (float): Time step for each simulation step.
        """
        logging.info(f"Initializing simulation with {num_particulas} particles.")
        self.posiciones = np.random.rand(num_particulas, 2) * np.array([ancho, alto])*2 - np.array([ancho, alto])
        self.velocidades = np.zeros((num_particulas, 2))
        self.radio_particula = radio_particula
        self.N = 20
        self.contador = 0
        self.frames = []
        self.dt = 0.01

    def mover_particulas(self, movimiento_maximo: float = 0.5) -> None:
        """ 
        Moves the particles based on their velocities and time step. 
        
        Parameters:
        - movimiento_maximo (float, optional): Maximum random movement. Default is 0.5.
        """
        self.posiciones += self.velocidades * self.dt
        self.corregir_colisiones()

    @cronometrar
    def corregir_colisiones(self) -> None:
        """ 
        Corrects the particle positions if they collide with each other or with the boundaries.
        
        This function uses a KDTree for efficient spatial queries. The time complexity is 
        approximately O(N*log(N)) for building the tree and O(N) for querying, where N is the number of particles.
        The space complexity is O(N) for storing the tree.
        """
        # Collisions with the right and left boundaries
        mask_x = np.where(self.posiciones[:, 0] > 1)
        self.posiciones[mask_x, 0] = 1 - (self.posiciones[mask_x, 0] - 1)
        self.velocidades[mask_x, 0] *= -1

        mask_x = np.where(self.posiciones[:, 0] < -1)
        self.posiciones[mask_x, 0] = -1 + (-1 - self.posiciones[mask_x, 0])
        self.velocidades[mask_x, 0] *= -1

        # Collisions with the top and bottom boundaries
        mask_y = np.where(self.posiciones[:, 1] > 1)
        self.posiciones[mask_y, 1] = 1 - (self.posiciones[mask_y, 1] - 1)
        self.velocidades[mask_y, 1] *= -1

        mask_y = np.where(self.posiciones[:, 1] < -1)
        self.posiciones[mask_y, 1] = -1 + (-1 - self.posiciones[mask_y, 1])
        self.velocidades[mask_y, 1] *= -1

        # Inter-particle collisions
        tree = cKDTree(self.posiciones)
        colisiones = tree.query_pairs(2 * self.radio_particula)
        for i, j in colisiones:
            # Calculate the direction vector
            direccion = self.posiciones[j] - self.posiciones[i]
            distancia = np.linalg.norm(direccion)
            overlap = 2 * self.radio_particula - distancia
            correccion = overlap / 2 * (direccion / distancia)
            self.posiciones[i] -= correccion
            self.posiciones[j] += correccion

    def actualizar_velocidades(self, n: int, m: int) -> None:
        """ 
        Updates the velocities of the particles based on the amplitude function.
        
        Parameters:
        - n (int): Harmonic order in x direction.
        - m (int): Harmonic order in y direction.
        
        The time complexity is O(N) where N is the number of particles, 
        since it involves array operations for each particle. 
        The space complexity is also O(N) for storing the new velocities.
        """
        magnitudes = amplitude(self.posiciones[:, 0], self.posiciones[:, 1], n, m)
        angulos = 2 * np.pi * np.random.rand(self.posiciones.shape[0])
        velocidades_x = magnitudes * np.cos(angulos)
        velocidades_y = magnitudes * np.sin(angulos)
        self.velocidades = np.column_stack((velocidades_x, velocidades_y))
        self.velocidades += (np.random.rand(*self.posiciones.shape) - 0.5) * 0.05

    @cronometrar
    def step(self, n: int, m: int) -> None:
        """ 
        Moves the simulation one step forward. 
        
        Parameters:
        - n (int): Harmonic order in x direction for the amplitude function.
        - m (int): Harmonic order in y direction for the amplitude function.
        """
        self.mover_particulas()
        self.contador += 1
        if self.contador % self.N == 0:
            logging.info("Updating particle velocities.")
            self.actualizar_velocidades(n, m)

    def plot_fondo(self, ax, n: int, m: int, res=100) -> None:
        """ 
        Plots the background amplitude pattern on a given axis.
        
        Parameters:
        - ax (matplotlib axis): Axis on which to plot.
        - n (int): Harmonic order in x direction for the amplitude function.
        - m (int): Harmonic order in y direction for the amplitude function.
        - res (int, optional): Resolution of the plot. Default is 100.
        
        The time complexity of this function is O(res^2) due to the meshgrid operation 
        and the space complexity is also O(res^2) for storing the grid values.
        """
        x = np.linspace(-1, 1, res)
        y = np.linspace(-1, 1, res)
        X, Y = np.meshgrid(x, y)
        Z = amplitude(X, Y, n, m)
        ax.imshow(Z, extent=[-1, 1, -1, 1], origin='lower', cmap='magma', aspect='auto')

    def init_animation(self):
        """ 
        Initializes the animation by plotting the background and particle positions.
        
        Returns:
        - tuple: The scatter plot of particle positions.
        
        This function has a time and space complexity of O(N), where N is the number of particles.
        """
        self.plot_fondo(self.ax, self.n, self.m)
        self.scatter = self.ax.scatter(self.posiciones[:, 0], self.posiciones[:, 1], s=1, c='white')
        return (self.scatter,)

    def update_animation(self, frame):
        """ 
        Updates the animation for a given frame.
        
        Parameters:
        - frame (int): The current frame number.
        
        Returns:
        - tuple: The scatter plot of updated particle positions.
        
        This function has a time complexity of O(N) for updating positions, 
        and a space complexity of O(N) for storing the new positions.
        """
        self.ax.clear()
        self.plot_fondo(self.ax, self.n, self.m)
        self.step(self.n, self.m)
        self.scatter = self.ax.scatter(self.posiciones[:, 0], self.posiciones[:, 1], s=1, c='white')
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis('off')
        return (self.scatter,)

    def run_simulation(self, n: int, m: int, num_frames: int, save_mp4=True):
        """ 
        Executes the simulation and displays the animation.
        
        Parameters:
        - n (int): Harmonic order in x direction for the amplitude function.
        - m (int): Harmonic order in y direction for the amplitude function.
        - num_frames (int): Total number of frames for the animation.
        - save_mp4 (bool, optional): If True, saves the animation as an mp4 file. Default is True.
        
        This function's complexity is dominated by the number of frames, 
        so the time complexity is O(num_frames * N) with N being the number of particles. 
        The space complexity is O(N) for storing particle positions and velocities.
        """
        self.n, self.m = n, m
        self.fig, self.ax = plt.subplots(figsize=(7, 7))
        self.ax.set_xlim(-1, 1)
        self.ax.set_ylim(-1, 1)
        self.ax.axis('off')
        anim = FuncAnimation(self.fig, self.update_animation, frames=num_frames, init_func=self.init_animation, blit=True)
        if save_mp4:
            anim.save('dinamica.mp4', writer='ffmpeg', fps=60)
        plt.show()

    def configurar(self, num_particulas, radio_particula, n, m):
        """Reconfigures the simulation according to user interface values."""
        '''
        configurar: This function reconfigures the simulation based on user-defined values.
        Inputs:
        num_particulas (int): Number of particles.
        radio_particula (float): Radius of each particle.
        n (int): Value of n for the amplitude function.
        m (float): Value of m for the amplitude function.
        No Outputs. It modifies the internal state of the simulation object.

        configurar:
        Time complexity: O(n)O(n), where nn is the number of particles. This is due to the initialization of 
        particle positions and velocities.
        Space complexity: O(n)O(n), as it stores the positions and velocities for all particles.
        '''

        # Initialize particle positions randomly within the space.
        self.posiciones = np.random.rand(num_particulas, 2) * np.array([1, 1]) * 2 - np.array([1, 1])
        
        # Initialize velocities to zero for all particles.
        self.velocidades = np.zeros((num_particulas, 2))
        
        # Set the particle radius.
        self.radio_particula = radio_particula
        
        # Store the values of n and m for the amplitude function.
        self.n = n
        self.m = m

    def run_pygame_visualization(self):
        """Runs the particle simulation visualization using pygame."""
        '''
        run_pygame_visualization: This function visualizes the particle simulation using the pygame library.
        No Inputs (except the object's internal state).
        No Outputs. It runs an interactive simulation window.

        run_pygame_visualization:
        Time complexity: Within the main loop, the step function's complexity is executed in each iteration, 
        plus the loop iterating through each particle to draw it, making it O(n)O(n), where nn is the number 
        of particles. The loop runs for as long as the simulation is active.
        Space complexity: O(n)O(n) due to the storage of particle positions.
        '''
        # Initialize pygame.
        pygame.init()
        
        # Set up the display screen.
        screen = pygame.display.set_mode((800, 800))

        # Calculate particle radius for visualization. 
        # Ensure it has a minimum value of 1 pixel.
        radio_particula = max([1, int(self.radio_particula * 400)])

        # Main loop to keep the visualization running.
        running = True
        while running:
            # Check for events, like window close.
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Update particle positions and velocities.
            self.step(self.n, self.m)

            # Clear the screen and draw particles in their new positions.
            screen.fill((0, 0, 0))
            for pos in self.posiciones:
                pygame.draw.circle(screen, (255, 255, 255), (int(pos[0] * 400 + 400), int(pos[1] * 400 + 400)), radio_particula)

            # Update the display.
            pygame.display.flip()

        # Exit pygame once the main loop ends.
        pygame.quit()

# Function to compute amplitude
def amplitude(xv: float, yv: float, n: int, m: int) -> float: 
    """ 
    Computes the amplitude function based on given harmonic orders and positions.
    
    Parameters:
    - xv (float): x-coordinate.
    - yv (float): y-coordinate.
    - n (int): Harmonic order in x direction.
    - m (int): Harmonic order in y direction.
    
    Returns:
    - float: Amplitude value at the given position.
    
    The time complexity is O(1) as it involves a constant number of operations, 
    and the space complexity is O(1) as well.
    """
    return np.abs(np.sin(n*np.pi*xv/2)*np.sin(m*np.pi*yv/2)-np.sin(m*np.pi*xv/2)*np.sin(n*np.pi*yv/2)) * 2

class TestSimulacion(unittest.TestCase):

    def setUp(self):
        """Setup function called before each test in this class."""
        self.sim = Simulacion(1000, 1, 1, 0.01)
    
    def test_inicializacion(self):
        """Prueba si la inicialización se realiza correctamente."""
        self.assertEqual(self.sim.posiciones.shape, (1000, 2))
        self.assertEqual(self.sim.velocidades.shape, (1000, 2))
        self.assertEqual(self.sim.radio_particula, 0.01)
        
    def test_mover_particulas(self):
        """Test the movement of particles."""
        posiciones_iniciales = self.sim.posiciones.copy()
        self.sim.mover_particulas()
        self.assertFalse(np.array_equal(self.sim.posiciones, posiciones_iniciales))

    def test_actualizar_velocidades(self):
        """Test the velocity update function."""

        velocidades_iniciales = self.sim.velocidades.copy()
        self.sim.actualizar_velocidades(5, 2)
        self.assertFalse(np.array_equal(self.sim.velocidades, velocidades_iniciales))
        
    def test_corregir_colisiones(self):
        """Test the collision correction function."""
        # Movemos todas las partículas fuera del borde para forzar una colisión
        self.sim.posiciones += 0.1
        self.sim.corregir_colisiones()
        # Después de corregir, todas las partículas deben estar dentro de los límites
        self.assertTrue(((self.sim.posiciones >= -1) & (self.sim.posiciones <= 1)).all())

class Configuracion(tk.Tk):
    def __init__(self, sim):
        """Initializes the GUI configuration window."""
        super().__init__()
        self.sim = sim

        ttk.Label(self, text="Número de partículas:").grid(column=0, row=0)
        self.particles = ttk.Entry(self)
        self.particles.grid(column=1, row=0)
        self.particles.insert(0, "15000")

        ttk.Label(self, text="Radio de partículas:").grid(column=0, row=1)
        self.radius = ttk.Entry(self)
        self.radius.grid(column=1, row=1)
        self.radius.insert(0, "0.003")


        ttk.Label(self, text="n:").grid(column=0, row=2)
        self.n = ttk.Entry(self)
        self.n.grid(column=1, row=2)
        self.n.insert(0, "3")


        ttk.Label(self, text="m:").grid(column=0, row=3)
        self.m = ttk.Entry(self)
        self.m.grid(column=1, row=3)
        self.m.insert(0, "7")

        # ... (otros campos de configuración) ...

        self.start_button = ttk.Button(self, text="Iniciar Simulación", command=self.iniciar_simulacion)
        self.start_button.grid(columnspan=2)

    def iniciar_simulacion(self):
        """Starts the simulation based on the user's input."""
        
        # Obtener valores de los campos
        num_particles = int(self.particles.get())
        rad = float(self.radius.get())
        n = float(self.n.get())
        m = float(self.m.get())

        # Configurar simulación
        self.sim.configurar(num_particles, rad, n, m)
        
        # Llamar a la visualización con pygame
        self.sim.run_pygame_visualization()

        self.destroy()

def main():
    '''
    As for the Big O notation:
    setUp: O(1)O(1) - constant time. This function only initializes the simulation object.
    test_*: O(1)O(1) to O(n)O(n), where nn is the number of particles. These functions test various functionalities, and their complexity depends on the specific test.
    __init__: O(1)O(1) - constant time. This function only initializes GUI components.
    iniciar_simulacion: O(1)O(1) - constant time. It reads user input and starts the simulation.
    main: O(1)O(1) - constant time. This function parses command-line arguments and decides which action to take.
    The spatial complexity of these functions generally depends on the number of particles and the size of the simulation space, but these complexities are predominantly linear, O(n)O(n), with respect to the number of particles.
    '''

    parser = argparse.ArgumentParser(description="Ejecuta una simulación de partículas en movimiento.")
    
    # Argumento para decidir qué acción realizar
    parser.add_argument("--action", choices=["gui", "test", "sim"], default="gui", help="Acción a realizar: gui para la interfaz gráfica, test para pruebas, sim para simulación directa.")
    
    # Resto de argumentos
    parser.add_argument("-p", "--particles", type=int, default=1000, help="Número de partículas en la simulación.")
    parser.add_argument("-r", "--radius", type=float, default=0.01, help="Radio de cada partícula.")
    parser.add_argument("-f", "--frames", type=int, default=100, help="Número de frames en la simulación.")
    parser.add_argument("-n", "--nvalue", type=int, default=5, help="Valor de n para la función de amplitud.")
    parser.add_argument("-m", "--mvalue", type=float, default=2, help="Valor de m para la función de amplitud.")
    parser.add_argument("--save_mp4", action="store_true", help="Guardar la simulación como un archivo .mp4.")
    
    args = parser.parse_args()

    # Acción condicional basada en el argumento --action
    if args.action == "gui":
        sim = Simulacion(15000, 1, 1, 0.005)
        app = Configuracion(sim)
        app.mainloop()
    elif args.action == "test":
        unittest.main(argv=['first-arg-is-ignored'], exit=False)

    elif args.action == "sim":
        sim = Simulacion(args.particles, 1, 1, args.radius)
        sim.run_simulation(args.nvalue, args.mvalue, args.frames, args.save_mp4)

if __name__ == '__main__':
    main()



'''
### Explicación y Complejidad de KD-Tree:

Un KD-Tree (K-dimensional tree) es una estructura de datos de árbol utilizada para organizar puntos en un espacio k-dimensional. KD-Tree es útil para aplicaciones que involucran búsquedas multidimensionales, como búsqueda de vecinos más cercanos.

#### ¿Cómo funciona?

- **Construcción**:
  1. Se selecciona un eje (generalmente se alterna entre los ejes disponibles) y se encuentra la mediana de los puntos basada en ese eje.
  2. Se divide el conjunto de puntos en dos subconjuntos: puntos a la izquierda y a la derecha de la mediana.
  3. Se repite el proceso de forma recursiva para cada subconjunto, utilizando un eje diferente en cada nivel.

- **Consulta**:
  1. Al buscar puntos dentro de un radio o vecinos más cercanos, el algoritmo navega por el árbol, eliminando rápidamente áreas del espacio que no pueden contener puntos relevantes.

#### Complejidad:

- **Construcción**:
  - Tiempo: \(O(n \log^2 n)\) en promedio, pero puede llegar a \(O(n^2)\) en el peor de los casos.
  - Espacio: \(O(n)\)

- **Consulta (búsqueda de vecinos más cercanos)**:
  - Tiempo: \(O(\log n)\) en promedio, pero puede llegar a \(O(n)\) en el peor de los casos para puntos que no están balanceados. En la práctica, la búsqueda es muy rápida para conjuntos de datos razonablemente distribuidos.
  - Espacio: \(O(\log n)\) para la pila de recursión.

#### Matemática detrás del KD-Tree:

La idea detrás del KD-Tree se basa en la partición del espacio. Al dividir repetidamente el espacio en función de las medianas de los puntos, el KD-Tree logra organizar los datos de manera que las consultas espaciales puedan realizarse de manera eficiente. Las particiones se representan mediante hiperplanos perpendiculares a uno de los ejes del espacio. Al navegar por el árbol durante una consulta, el algoritmo puede determinar rápidamente si necesita explorar un subárbol o puede descartarlo por completo.

El KD-Tree es eficiente en dimensiones bajas a moderadas (digamos, hasta 20 dimensiones). Sin embargo, su eficiencia disminuye a medida que aumenta la dimensionalidad del espacio, un fenómeno conocido como "maldición de la dimensionalidad".

En resumen, KD-Tree es una herramienta poderosa para búsquedas espaciales rápidas en conjuntos de datos con dimensiones bajas a moderadas, y `scipy.spatial.cKDTree` ofrece una implementación eficiente para Python.
'''
