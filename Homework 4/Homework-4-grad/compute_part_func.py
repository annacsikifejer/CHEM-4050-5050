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



temperatures = np.linspace(10, 500, 100)                                    # Defines temperature range from 10 K to 500 K with increments of 100 K

partition_functions = []                                                    # Creates a list to store partition functions
for temp in temperatures:                                                   # Calls each temperature value from temperature range
    z = compute_partition_function(temp)                                    # Computes partition function at a given temperature value
    partition_functions.append(z)                                           # Appends partition function to list

partition_functions = np.array(partition_functions)                         # Creates a NumPy array of partition function values

partition_function_df = pd.DataFrame({                                      # Creates a dataframe of partition function vs temperature
    'Temperature': temperatures,                                            # Temperature values in dataframe taken from temperature values
    'Partition Function': partition_functions                               # Partition function values in dataframe taken from partition function values
})

partition_function_df.to_csv('partition_functions_with_temperature.csv', index = False)                    # Saves partition function dataframe as a csv file
print("CSV file 'partition_functions_with_temperature.csv' created successfully.")                         # Prints success message
