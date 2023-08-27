# Particle Resonance Simulation

This program simulates the resonance of a vibrating 2D plate with particles (like sand) on it, giving a visual representation of eigenmodes of vibration.

## Table of Contents
- [Physics Behind the Simulation](#physics-behind-the-simulation)
- [Complexity Analysis](#complexity-analysis)
- [Running the Simulation](#running-the-simulation)
- [Dependencies](#dependencies)
- [Contributions](#contributions)

## Physics Behind the Simulation

For a vibrating 2D plate, let the height at some point on the plate during vibration be \( u(x,y,t) \). The motion of this plate is governed by the differential equation:

 $u_{tt} = c^2\nabla^2 u $

where $ c $ represents the wave speed.

The eigenmodes of vibration, which represent the specific frequencies at which the plate vibrates, are determined by:

$ f(n,m) = \frac{c}{2\pi} \sqrt{\frac{n^2}{L_x^2} + \frac{m^2}{L_y^2}} $

Here, $ L_x $ and $ L_y $ are the dimensions of the rectangular plate, while $ n$  and $ m $ are integers. If the values for $ c $, $ L_x $, and $ L_y $ are known, these special frequencies can be determined by plugging in different integer values for $n$ and $ m $.

At these particular frequencies, the solution in space and time is separable, with $ u(x,y,t) = U(x,y)G(t) $. Under certain conditions, the maximum amplitude occurs at the boundary of the plates, and:

$ U(x,y) \propto |\sin(n \pi x/L_x)\sin(m \pi y/L_y) - \sin(m \pi x/L_x)\sin(n \pi y/L_y)| $

where $ n $ and $ m $ are **odd** integers corresponding to a particular driving frequency that results in a specific mode of vibration.

## Complexity Analysis

**Spatial and Temporal Complexity:**

- The main simulation involves a KD-Tree (K-dimensional tree) for efficient spatial queries. The time complexity is approximately $ O(N \log^2 N) $ for building the tree and $ O(N) $ for querying, where $N $ is the number of particles. The space complexity is $ O(N) $ for storing the tree.
- Other functions have complexities ranging from $ O(1) $ (constant time) to $ O(N)$, where $ N $ represents the number of particles.

## Running the Simulation

- Use the GUI for a more user-friendly experience and to adjust parameters.
- The simulation can be executed directly with specific parameters.
- Tests are included to ensure functionality.

## Dependencies

- numpy
- matplotlib
- scipy
- imageio
- pygame
- tkinter
- unittest
- argparse

## Contributions

Contributions are welcome! Please ensure that you update tests as appropriate and keep the physics of the simulation intact.
