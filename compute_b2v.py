import numpy as np                                                                 # Imports the NumPy library and saves as np
import pandas as pd                                                                # Imports the Pandas library and saves as pd
import scipy.integrate                                                             # Imports the Integrate submodule of the SciPy library
import matplotlib.pyplot as plt                                                    # Imports the Pyplot submodule of the MatPlot Library and saves as plt


def hard_sphere_potential(r, sigma = 3.4):                                         # Defines the hard-sphere potential

    return np.where(r < sigma, np.inf, 0)

def square_well_potential(r, sigma = 3.4, epsilon = 0.01, lambda_sigma = 1.5):     # Defines the square-well potential

    return np.where(r < sigma, np.inf, np.where(r < lambda_sigma * sigma, -epsilon, 0))

def lennard_jones_potential(r, epsilon = 0.01, sigma = 3.4):                       # Defines the Lennard-Jones potential
    repulsive_term = (sigma / r)**12                                               # Repulsive potential between argon atoms
    attractive_term = (sigma / r)**6                                               # Attractive force between argon atoms
    return 4 * epsilon * (repulsive_term - attractive_term)                        # Lennard-Jones potential


T_100K = 100                                                                       # Defines initial temperature T_100K as 100 K
kB = 1.380649e-23                                                                  # Defines Boltzmann's constant (J/K)
N_A = 6.022e23                                                                     # Defines # Avogadro's number
conversion_factor = 1e6                                                            # Defines conversion factor between m cubed and cm cubed

sigma = 3.4e-10                                                                    # sigma is the diameter for the hard-sphere model, the particle diameter for the square-well model, and the distance at which the potential is zero for the Lennard-Jones potential
epsilon_J = 0.01 * 1.60218e-19                                                     # epsilon is the depth of the well
lambda_sigma = 1.5                                                                 # lambda is the range of the well

def integrand(r, T, potential_func, *args):                                        # Defines b2v equation

    potential = potential_func(r, *args)
    if np.isinf(potential):
        return -r**2
    else:
        return r**2 * (np.exp(-potential / (kB * T)) - 1)


r_start_hs = 1e-12                                        
r_end_hs = sigma
integral_result_hs_100K, error_hs_100K = scipy.integrate.quad(integrand, r_start_hs, r_end_hs, args = (T_100K, hard_sphere_potential, sigma))
B2_hard_sphere_100K = -2 * np.pi * integral_result_hs_100K                          # Calculates the hard-sphere potential at 100 K
B2_hard_sphere_cm3_mol_100K = B2_hard_sphere_100K * N_A * conversion_factor         # Converts units to cm cubed / mol

r_start_sw = 1e-12
r_end_sw = lambda_sigma * sigma
integral_result_sw_100K, error_sw_100K = scipy.integrate.quad(integrand, r_start_sw, r_end_sw, args=(T_100K, square_well_potential, sigma, epsilon_J, lambda_sigma))
B2_square_well_100K = -2 * np.pi * integral_result_sw_100K                          # Calculates the square-well potential at 100 K
B2_square_well_cm3_mol_100K = B2_square_well_100K * N_A * conversion_factor         # Converts units to cm cubed / mol

r_start_lj = 1e-12
r_end_lj = 5 * sigma # Sufficiently large distance
integral_result_lj_100K, error_lj_100K = scipy.integrate.quad(integrand, r_start_lj, r_end_lj, args=(T_100K, lennard_jones_potential, epsilon_J, sigma))
B2_lennard_jones_100K = -2 * np.pi * integral_result_lj_100K                        # Calculates the Lennard-Jones potential at 100 K
B2_lennard_jones_cm3_mol_100K = B2_lennard_jones_100K * N_A * conversion_factor     # Converts units to cm cubed / mol

print(f"B2v for Hard-Sphere potential at {T_100K} K = {B2_hard_sphere_cm3_mol_100K:.2f} cm³/mol")        # Prints b2v for hard-sphere potential at 100K to 2 decimal places
print(f"B2v for Square-Well potential at {T_100K} K = {B2_square_well_cm3_mol_100K:.2f} cm³/mol")        # Prints b2v for square-well potential at 100K to 2 decimal places
print(f"B2v for Lennard-Jones potential at {T_100K} K = {B2_lennard_jones_cm3_mol_100K:.2f} cm³/mol")    # Prints b2v for Lennard-Jones potential at 100K to 2 decimal places


temperatures_k = np.linspace(100, 800, 50)                               # Defines a range of temperatures from 100 K to 800 K with 50-K increments

b2v_hard_sphere_values = []                                              # Creates a list for hard-sphere b2v values
b2v_square_well_values = []                                              # Creates a list for square-well b2v values
b2v_lennard_jones_values = []                                            # Creates a list for Lennard-Jones b2v values

for T in temperatures_k:
    integral_result_hs, error_hs = scipy.integrate.quad(integrand, r_start_hs, r_end_hs, args = (T, hard_sphere_potential, sigma))                             # Calculates hard-sphere potential
    B2_hard_sphere = -2 * np.pi * integral_result_hs
    b2v_hard_sphere_values.append(B2_hard_sphere * N_A * conversion_factor)       # Appends hard-sphere potential to list

    integral_result_sw, error_sw = scipy.integrate.quad(integrand, r_start_sw, r_end_sw, args = (T, square_well_potential, sigma, epsilon_J, lambda_sigma))    # Calculates square-well potential
    B2_square_well = -2 * np.pi * integral_result_sw
    b2v_square_well_values.append(B2_square_well * N_A * conversion_factor)       # Appends square-well potential to list

    integral_result_lj, error_lj = scipy.integrate.quad(integrand, r_start_lj, r_end_lj, args = (T, lennard_jones_potential, epsilon_J, sigma))                # Calculates Lennard-Jones potential
    B2_lennard_jones = -2 * np.pi * integral_result_lj
    b2v_lennard_jones_values.append(B2_lennard_jones * N_A * conversion_factor)   # Appends Lennard-Jones potential to list


plt.figure(figsize = (10, 6))                                                     # Creates a plot
plt.plot(temperatures_k, b2v_hard_sphere_values, label = 'Hard-Sphere')           # Plots hard-sphere data
plt.plot(temperatures_k, b2v_square_well_values, label = 'Square-Well')           # Plots square-well data
plt.plot(temperatures_k, b2v_lennard_jones_values, label = 'Lennard-Jones')       # Plots Lennard-Jones data
plt.xlabel("Temperature (K)")                                                     # Creates label for x-axis
plt.ylabel("Second Virial Coefficient (cm³/mol)")                                 # Creates label for y-axis
plt.title("Second Virial Coefficient vs. Temperature for Different Potentials")   # Creates title
plt.legend()                                                                      # Shows legend
plt.axhline(y = 0, color = 'grey', linestyle = '--')                              # Creates a horizontal line at b2v = 0
plt.grid(True)                                                                    # Shows gridlines
plt.show()                                                                        # Prints plot


df_b2v = pd.DataFrame(b2v_lennard_jones_values)                                   # Creates a dataframe of Lennard-Jones potentials
df_b2v.to_csv("b2v_data.csv", index = False)                                      # Saves data as csv file

print("Data saved to b2v_data.csv")                                               # Prints confirmation of csv file save