import numpy as np                                          # Imports the NumPy library and saves as np
import matplotlib.pyplot as plt                             # Imports the Pyplot submodule of the MatPlot Library and saves as plt

N = 100                                                     # Number of data points
L = 10                                                      # Length of observation
dx = L / N                                                  # Spacing between data points
x = np.linspace(-L/2, L/2, N)                               # Defines the real-space grid (L = 40 au)
m = 1                                                       # Defines mass of oscillator
hbar = 1                                                    # Defines reduced Planck's constant

def create_potential_matrix(x, k, lambda_val = 0):          # Defines potential matrix

    potential = 0.5 * k * x**2 + lambda_val * x**4
    return np.diag(potential)

def create_laplacian_matrix(N, dx, hbar, m):                # Defines laplacian matrix

    laplacian = (hbar**2 / (2 * m * dx**2)) * (np.diag(-2 * np.ones(N)) +
                                             np.diag(np.ones(N - 1), k = 1) +
                                             np.diag(np.ones(N - 1), k = -1))
    return laplacian

def create_hamiltonian_matrix(laplacian_matrix, potential_matrix):    # Defines Hamiltonian matrix

    return potential_matrix - laplacian_matrix


k = 1.0                                                  # Defines the harmonic potential constant
lambda_val = 0.1                                         # Defines the anharmonic potential constant

potential_harmonic = create_potential_matrix(x, k)                                           # Creates the potential harmonic matrix   
laplacian = create_laplacian_matrix(N, dx, hbar, m)                                          # Creates the laplacian matrix 
hamiltonian_harmonic = create_hamiltonian_matrix(laplacian, potential_harmonic)              # Creates the hamiltonian harmonic matrix 

eigenvalues_harmonic, eigenfunctions_harmonic = np.linalg.eigh(hamiltonian_harmonic)         # Calculates the harmonic eigenfunctions and eigenvalues


potential_anharmonic = create_potential_matrix(x, k, lambda_val)                             # Creates the potential anharmonic matrix 
hamiltonian_anharmonic = create_hamiltonian_matrix(laplacian, potential_anharmonic)          # Creates the hamiltonian anharmonic matrix 

eigenvalues_anharmonic, eigenfunctions_anharmonic = np.linalg.eigh(hamiltonian_anharmonic)   # Calculates the harmonic eigenfunctions and eigenvalues


sort_indices_harmonic = np.argsort(eigenvalues_harmonic)                                      # Sorts the harmonic eigenvalues and eigenfunctions
eigenvalues_harmonic = eigenvalues_harmonic[sort_indices_harmonic]
eigenfunctions_harmonic = eigenfunctions_harmonic[:, sort_indices_harmonic]

sort_indices_anharmonic = np.argsort(eigenvalues_anharmonic)                                  # Sorts the anharmonic eigenvalues and eigenfunctions
eigenvalues_anharmonic = eigenvalues_anharmonic[sort_indices_anharmonic]
eigenfunctions_anharmonic = eigenfunctions_anharmonic[:, sort_indices_anharmonic]

fig_harmonic, axes_harmonic = plt.subplots(10, 1, figsize = (8, 20))
fig_harmonic.suptitle('Harmonic Potential - First 10 Wavefunctions', y = 1.02)                 # Creates a section title

for i in range(10):
    axes_harmonic[i].plot(x, eigenfunctions_harmonic[:, i])                                    # Plots the first 10 energy levels of the harmonic potential
    axes_harmonic[i].set_title(f'Energy Level: {eigenvalues_harmonic[i]:.3f}')                 # Creates a plot title
    axes_harmonic[i].set_xlabel('Position')                                                    # Creates label for x-axis
    axes_harmonic[i].set_ylabel('Wavefunction Amplitude')                                      # Creates label for y-axis
    axes_harmonic[i].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)

plt.tight_layout()                                                                             # Combines the plot layouts to look nicer

fig_anharmonic, axes_anharmonic = plt.subplots(10, 1, figsize = (8, 20))
fig_anharmonic.suptitle('Anharmonic Potential - First 10 Wavefunctions', y = 1.02)             # Creates a section title

for i in range(10):
    axes_anharmonic[i].plot(x, eigenfunctions_anharmonic[:, i])                                 # Plots the first 10 energy levels of the anharmonic potential
    axes_anharmonic[i].set_title(f'Energy Level: {eigenvalues_anharmonic[i]:.3f}')              # Creates a plot title
    axes_anharmonic[i].set_xlabel('Position')                                                   # Creates label for x-axis
    axes_anharmonic[i].set_ylabel('Wavefunction Amplitude')                                     # Creates label for y-axis
    axes_anharmonic[i].axhline(0, color = 'gray', linestyle = '--', linewidth = 0.8)

plt.tight_layout()                                                                               # Combines the plot layouts to look nicer

plt.show()                                                                                       # Prints plot
