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


def total_lennard_jones_potential(coords, epsilon = 0.01, sigma = 3.4):
    
    r12, x3, y3 = coords                                                           # Coordinates to be optimized

    atom1 = np.array([0, 0])                                                       # Atom 1 is fixed at (0,0)
    atom2 = np.array([r12, 0])                                                     # Atom 2 is fixed along y = 0
    atom3 = np.array([x3, y3])

    r13 = np.linalg.norm(atom3 - atom1)                                            # Calculates distance from Atom 1 to Atom 3
    r23 = np.linalg.norm(atom3 - atom2)                                            # Calculates distance from Atom 2 to Atom 3

    v12 = lennard_jones_potential(r12, epsilon, sigma)                             # Lennard-Jones potential between Atom 1 and Atom 2
    v13 = lennard_jones_potential(r13, epsilon, sigma)                             # Lennard-Jones potential between Atom 1 and Atom 3
    v23 = lennard_jones_potential(r23, epsilon, sigma)                             # Lennard-Jones potential between Atom 2 and Atom 3

    v_total = v12 + v13 + v23                                                      # Total potential energy is equal to the sum of the potential energies of each argon atom

    return v_total


r12_initial = min_distance                                                         # Minimum distance found for argon dimer is used as intial coordinates for optimization

x3_initial = r12_initial / 2                                                       # Initial guess of geometry as an equalateral triangle to allow for intial coordinate guess
y3_initial = np.sqrt(3) / 2 * r12_initial
initial_coords = np.array([r12_initial, x3_initial, y3_initial])                   # Intial coordinate values

optimization_result = scipy.optimize.minimize(total_lennard_jones_potential, initial_coords, method = 'Nelder-Mead')   # Optimize with SciPy using the Nelder-Mead Method

optimized_coords = optimization_result.x                                           # Determines coordinates that optimize the potential energy for each Atom
minimized_potential_energy = optimization_result.fun

optimized_r12, optimized_x3, optimized_y3 = optimized_coords                       # Stores optimized coordinates

atom1_optimized = np.array([0, 0])                                                 # Stores Atom 1 coordinates in an array
atom2_optimized = np.array([optimized_r12, 0])                                     # Stores Atom 2 coordinates in an array
atom3_optimized = np.array([optimized_x3, optimized_y3])                           # Stores Atom 3 coordinates in an array


plt.figure(figsize = (6, 6))                                                                       # Creates a scatterplot of each argon atom relative to one another
plt.scatter(atom1_optimized[0], atom1_optimized[1], color = 'black', marker = 'o', s = 100)        # Creates a black dot at the coordinates of Atom 1
plt.scatter(atom2_optimized[0], atom2_optimized[1], color = 'black', marker = 'o', s = 100)        # Creates a black dot at the coordinates of Atom 2
plt.scatter(atom3_optimized[0], atom3_optimized[1], color = 'black', marker = 'o', s = 100)        # Creates a black dot at the coordinates of Atom 3

plt.plot([atom1_optimized[0], atom2_optimized[0]], [atom1_optimized[1], atom2_optimized[1]], color = 'black')   # Draws black line from Atom 1 to Atom 2
plt.plot([atom1_optimized[0], atom3_optimized[0]], [atom1_optimized[1], atom3_optimized[1]], color = 'black')   # Draws black line from Atom 1 to Atom 3
plt.plot([atom2_optimized[0], atom3_optimized[0]], [atom2_optimized[1], atom3_optimized[1]], color = 'black')   # Draws black line from Atom 2 to Atom 3

plt.text(atom1_optimized[0] + 0.7, atom1_optimized[1] - 0.3, 'Atom 1 (0,0)', ha = 'right', va = 'bottom')                       # Labels Atom 1 (below and to the right of dot) 
plt.text(atom2_optimized[0] - 1.0, atom2_optimized[1] - 0.3, f'Atom 2 ({optimized_r12:.3f}, 0)', ha = 'left', va = 'bottom')    # Labels Atom 2 (below and to the left of dot) 
plt.text(2.0, atom3_optimized[1] + 0.1, f'Atom 3 ({optimized_x3:.3f}, {optimized_y3:.3f})', ha = 'center', va = 'bottom')       # Labels Atom 3 (above and centered above dot) 

plt.plot([0, optimized_x3], [optimized_y3, optimized_y3], color = 'grey', linestyle = '--', label = 'y₃')        # Adds a grey horizontal line from y-axis
plt.plot([optimized_x3, optimized_x3], [0, optimized_y3], color = 'grey', linestyle = '--', label = 'x₃')        # Adds a grey vertical line from x-axis

plt.xlabel("x")                                              # Creates label for x-axis
plt.ylabel("y")                                              # Creates label for y-axis
plt.title("Optimized Geometry of Ar₃")                        # Creates title for plot
plt.axis('equal')                                            # Equal scaling makes it easier to see argon atom locations relative to each other
plt.grid(False)                                              # Gridlines make it harder to see argon atom locations relative to each other
plt.show()


r12_optimized = np.linalg.norm(atom2_optimized - atom1_optimized)                    # Optimal distance between Atom 1 and Atom 2
r13_optimized = np.linalg.norm(atom3_optimized - atom1_optimized)                    # Optimal distance between Atom 1 and Atom 3
r23_optimized = np.linalg.norm(atom3_optimized - atom2_optimized)                    # Optimal distance between Atom 2 and Atom 3

optimal_distances = (r12_optimized, r13_optimized, r23_optimized)                                                                # Stores all three distances as a tuple

vector_12 = atom2_optimized - atom1_optimized                                                                                    # Vector from Atom 1 to Atom 2
vector_13 = atom3_optimized - atom1_optimized                                                                                    # Vector from Atom 1 to Atom 3
angle_123 = np.degrees(np.arccos(np.dot(vector_12, vector_13) / (np.linalg.norm(vector_12) * np.linalg.norm(vector_13))))        # Angle of Atom 1

vector_21 = atom1_optimized - atom2_optimized                                                                                    # Vector from Atom 2 to Atom 1
vector_23 = atom3_optimized - atom2_optimized                                                                                    # Vector from Atom 2 to Atom 3
angle_123_at_2 = np.degrees(np.arccos(np.dot(vector_21, vector_23) / (np.linalg.norm(vector_21) * np.linalg.norm(vector_23))))   # Angle of Atom 2

vector_31 = atom1_optimized - atom3_optimized                                                                                    # Vector from Atom 3 to Atom 1
vector_32 = atom2_optimized - atom3_optimized                                                                                    # Vector from Atom 3 to Atom 2
angle_123_at_3 = np.degrees(np.arccos(np.dot(vector_31, vector_32) / (np.linalg.norm(vector_31) * np.linalg.norm(vector_32))))    # Angle of Atom 3

optimal_angles = (angle_123, angle_123_at_2, angle_123_at_3)                                                                      # Stores all three angles as a tuple


print(f"Optimized coordinates (r12, x3, y3): ({optimized_coords[0]:.3f}, {optimized_coords[1]:.3f}, {optimized_coords[2]:.3f})")  # Prints the optimized coordinates r12, x3, and y3 to 3 decimal places
print(f"Minimum potential energy: {minimized_potential_energy:.3f}")                                                              # Prints the minimum potential energy to 3 decimal places
print(f"The optimized distances (Angstroms) are r12: {optimal_distances[0]:.3f}, r13: {optimal_distances[1]:.3f}, r23: {optimal_distances[2]:.3f}")           # Prints the optimized distance between each argon atom in Angstroms to 3 decimal places
print(f"The optimal angles between the Ar atoms are: Atom 1: {optimal_angles[0]:.3f}°, Atom 2: {optimal_angles[1]:.3f}°, Atom 3: {optimal_angles[2]:.3f}°")   # Prints the optimized angle between each argon atom in degrees to 3 decimal places
print(f"Given the equal distances between each atom and three 60 degree internal angles, the optimized geometry is an equilateral triangle")                  # The optimized geometry describes that of an equilateral triangle