import numpy as np                                                    # Imports the NumPy library and designates as np
import pandas as pd                                                   # Imports the Pandas library and designates as pd
from scipy.integrate import trapezoid                                 # Imports the trapezoid function from the SciPy integrate submodule of the SciPy library
import matplotlib.pyplot as plt                                       # Imports the pyplot submodule of the matplotlib module and designates as plt



epsilon = 0.0103                                                      # Defines epsilon (eV) as 0.0103
sigma = 3.405                                                         # Defines sigma (Angstroms) as 3.405
box_length = 10.0                                                     # Defines box length (Angstroms) as 10.0

def compute_partition_function(temperature):                          # Defines partition function
    lower_limit = 0.1                                                 # Defines lower limit of integration as slightly above zero (avoids dividing by zero)
    upper_limit = box_length                                          # Defines upper limit of integration as the box length
    num_points = 1000                                                 # Defines the number of integration points
    
    r_values = np.linspace(lower_limit, upper_limit, num_points)      # Creates an array of datapoints to be used for integration from lower limit to upper limit
    k = 8.617333262145e-5                                             # Defines Boltzmann's constant (eV / K)

    lj_potential = 4 * epsilon * ((sigma / (r_values + 1e-9))**12 - (sigma / (r_values + 1e-9))**6)        # Calculates the LJ potential at each point (1e-9 added to avoid dividing by zero)

    integrand = r_values**2 * np.exp(-lj_potential / (k * temperature))                      # Calculates the integrand
    partition_function = trapezoid(integrand, r_values)                     # Calculates the partition function
    return partition_function                                               # Returns partition function

    

def compute_internal_energy(partition_function, temperatures):        # Defines internal energy function
    k = 8.617333262145e-5                                             # Defines Boltzmann's constant (eV / K)

    log_z = np.log(partition_function + 1e-10)                        # Calculates log of partition function (1e-10 added to avoid dividing by zero)
    d_log_z_dt = np.gradient(log_z, temperatures)                     # Calculates the derivative of log of partition function at a given temperature using gradient function of NumPy
    internal_energy = k * temperatures**2 * d_log_z_dt                # Calculates the internal energy

    return internal_energy                                            # Returns the internal energy 

    

def compute_heat_capacity(internal_energy, temperatures):             # Defines heat capacity function
    d_u_dt = np.gradient(internal_energy, temperatures)                       # Calculates heat capacity using gradient function of NumPy

    return d_u_dt                                                             # Returns heat capacity 

    

temperatures = np.linspace(10, 500, 100)                                      # Defines temperature range from 10 K to 500 K with increments of 100 K

partition_functions = []                                                      # Creates a list to store partition functions
for temp in temperatures:                                                     # Calls each temperature value from temperature range
    z = compute_partition_function(temp)                                      # Calculates partition function at a given temperature value
    partition_functions.append(z)                                             # Appends partition function to list

partition_functions = np.array(partition_functions)                           # Creates a NumPy array of partition function values

internal_energy = compute_internal_energy(partition_functions, temperatures)  # Calculates internal energy at a given temperature
heat_capacity = compute_heat_capacity(internal_energy, temperatures)          # Calculates heat capacity at a given temperature



partition_function_df = pd.DataFrame({                                                                     # Creates a dataframe of partition function vs temperature
    'Temperature': temperatures,
    'Partition Function': partition_functions
})

partition_function_df.to_csv('partition_functions_with_temperature.csv', index = False)                    # Saves partition function dataframe as a csv file
print("CSV file 'partition_functions_with_temperature.csv' created successfully.")                         # Prints success message


internal_energy_heat_capacity_df = thermodynamic_df[['Temperature', 'Internal Energy', 'Heat Capacity']]   # Saves internal energy and heat capacity data as a dataframe
internal_energy_heat_capacity_df.to_csv('internal_energy_heat_capacity.csv', index = False)                # Saves dataframe as a csv file
print("CSV file 'internal_energy_heat_capacity.csv' created successfully.")                                # Prints success message



max_cv_index = np.argmax(heat_capacity)                                                         # Finds maximum heat capacity
dissociation_temperature = temperatures[max_cv_index]                                           # Finds temperature at max heat capacity
print(f"The atomization temperature of the LJ dimer is: {dissociation_temperature:.2f} K")      # Prints atomization temperature (temp at max heat capacity)

plt.figure(figsize = (10, 6))                                                                   # Creates a plot that is 10 inches by 6 inches
plt.plot(temperatures, heat_capacity)                                                           # Plots temperature as a function of heat capacity
plt.xlabel("Temperature (K)")                                                                   # Creates x-axis label
plt.ylabel("Heat Capacity (Cv)")                                                                # Creates y-axis label
plt.title("Heat Capacity vs. Temperature for Lennard-Jones Dimer")                              # Creates plot title
plt.grid(True)                                                                                  # Turns on gridlines
plt.show()                                                                                      # Shows plot
