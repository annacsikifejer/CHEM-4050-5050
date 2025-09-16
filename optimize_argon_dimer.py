import numpy as np                                                                 # Imports the NumPy library and saves as np
import scipy.optimize                                                              # Imports the Optimize submodule of the SciPy library
import matplotlib.pyplot as plt                                                    # Imports the Pyplot submodule of the MatPlot Library and saves as plt

def lennard_jones_potential(r, epsilon = 0.01, sigma = 3.4):                       # Defines the Lennard-Jones potential

    repulsive_term = (sigma / r)**12                                               # Repulsive potential between argon atoms
    attractive_term = (sigma / r)**6                                               # Attractive force between argon atoms

    V_r = 4 * epsilon * (repulsive_term - attractive_term)                         # Lennard-Jones potential equation

    return V_r

result = scipy.optimize.minimize(lennard_jones_potential, x0 = 4)                  # Initial guess of 4 Angstroms
min_distance = result.x[0]

r_values = np.linspace(3, 6, 100)                                                  # Range of distances is 3 to 6 Angstroms with 100 total data points
potential_values = lennard_jones_potential(r_values)                               # Calculates the potential energy at each distance

plt.plot(r_values, potential_values)                                               # Creates a plot of the Lennard-Jones potential over distance
plt.xlabel("Distance (r)")                                                         # Creates label for x-axis
plt.ylabel("Potential Energy (V(r))")                                              # Creates label for y-axis
plt.title("Lennard-Jones Potential of Ar$_2$")                                     # Creates title for plot
plt.grid(True)                                                                     # Shows gridlines
min_potential = np.min(potential_values)                                           # Defines minimum Lennard-Jones potential
plt.axhline(y = min_potential, color = 'black', linestyle = '--', label = f'Minimum Potential (V(r) = {min_potential:.3f})')   # Plots a black horizontal line at the minimum potential energy
plt.axvline(x = min_distance, color = 'grey', linestyle = '--', label = f'Minimum Distance (r = {min_distance:.3f})')          # Plots a grey vertical line at the distance at which the minimum potential energy is reached
plt.legend()                                                                       # Shows legend
plt.show()                                                                         # Shows plot

print(f"The equilibrium distance for Ar₂ occurs at {min_distance:.3f} Angstroms")   # Prints equilibrium distance between argon atoms of dimer to 3 decimal places