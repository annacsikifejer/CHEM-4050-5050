import numpy as np                                                    # Imports the NumPy library and designates as np
import pandas as pd                                                   # Imports the Pandas library and designates as pd
import scipy.constants as const                                       # Imports the constants submodule of the SciPy library
import matplotlib.pyplot as plt                                       # Imports the pyplot submodule of the matplotlib module and designates as plt



def partition_function_ce3_isolated(degeneracy):                      # Defines the partition function for the isolated Ce3+ ion
  return degeneracy                                                   # Returns partition function for 14-fold degeneracy (zero-energy)

def partition_function_ce3_soc(temperature, degeneracy_ground, energy_excited, degeneracy_excited):    # Defines the partition function for Ce3+ ion with Spin-Orbit-Coupling (SOC)
  kT = const.k * temperature                                                 # Defines the thermal energy (kT) as product of Boltzmann's constant (const.k) and temperature
  boltzmann_ground = degeneracy_ground * np.exp(0 / kT)                      # Defines the Boltzmann ground energy
  boltzmann_excited = degeneracy_excited * np.exp(-energy_excited / kT)      # Defines the Boltzmann excited energy
  return boltzmann_ground + boltzmann_excited                                # Returns partition function

def partition_function_ce3_soc_cfs(temperature, energy_levels, degeneracies):                                # Defines the partition function for Ce3+ ion with both SOC and Crystal Field Splitting (CFS)
  kT = const.k * temperature                                                                                 # Defines the thermal energy (kT) as product of Boltzmann's constant (const.k) and temperature
  boltzmann_factors = [degeneracies[i] * np.exp(-energy_levels[i] / kT) for i in range(len(energy_levels))]  # Defines the Boltzmann factors
  return sum(boltzmann_factors)                                                                              # Returns the partition function



degeneracy_case1 = 14                                              # Defines 14-fold degeneracy for isolated Ce3+ ion

degeneracy_ground_case2 = 6                                        # Defines 6 2F5/2 states for Ce3+ ion with SOC
energy_difference_eV_case2 = 0.28                                  # Defines 0.28 eV energy difference between 2F5/2 and 2F7/2 states
energy_difference_J_case2 = energy_difference_eV_case2 * const.e   # Converts energy difference from eV to J
degeneracy_excited_case2 = 8                                       # Defines 8 2F7/2 states for Ce3+ ion with SOC

energy_levels_J_case3 = [energy * const.e for energy in energy_levels_eV_case3]    # Converts energy levels from eV to J for Ce3+ ion with SOC and CSF

temperatures = np.linspace(300, 2000, 100)                 # Defines tempearture range from 300 K to 2000 K with increments of 100 K                          




Z_case1 = [partition_function_ce3_isolated(degeneracy_case1) for T in temperatures]           # Finds partition function for isolated Ce3+ ion
U_case1 = [0 for T in temperatures]                                                           # Finds internal energy for isolated Ce3+ ion
F_case1 = [-const.k * T * np.log(Z) for T, Z in zip(temperatures, Z_case1)]                   # Finds free energy for isolated Ce3+ ion
S_case1 = [(u - a) / T for u, a, T in zip(U_case1, F_case1, temperatures)]                    # Finds entropy for isolated Ce3+ ion

Z_case2 = [partition_function_ce3_soc(T, degeneracy_ground_case2, energy_difference_J_case2, degeneracy_excited_case2) for T in temperatures]                         # Finds partition function for Ce3+ ion with SOC
U_case2 = [(degeneracy_excited_case2 * energy_difference_J_case2 * np.exp(-energy_difference_J_case2 / (const.k * T))) / Z for T, Z in zip(temperatures, Z_case2)]    # Finds internal energy for Ce3+ ion with SOC
F_case2 = [-const.k * T * np.log(Z) for T, Z in zip(temperatures, Z_case2)]                   # Finds free energy for Ce3+ ion with SOC
S_case2 = [(u - a) / T for u, a, T in zip(U_case2, F_case2, temperatures)]                    # Finds entropy for Ce3+ ion wth SOC

Z_case3 = [partition_function_ce3_soc_cfs(T, energy_levels_J_case3, degeneracies_case3) for T in temperatures]                                                       # Finds partition function for Ce3+ ion with SOC and CSF
U_case3 = [sum([degeneracies_case3[i] * energy_levels_J_case3[i] * np.exp(-energy_levels_J_case3[i] / (const.k * T)) for i in range(len(energy_levels_J_case3))]) / Z for T, Z in zip(temperatures, Z_case3)]                                                                   # Finds internal energy for Ce3+ ion with SOC and CSF
F_case3 = [-const.k * T * np.log(Z) for T, Z in zip(temperatures, Z_case3)]                   # Finds free energy for Ce3+ ion with SOC and CSF
S_case3 = [(u - a) / T for u, a, T in zip(U_case3, F_case3, temperatures)]                    # Finds entropy for Ce3+ ion wth SOC and CSF

data = {                                                      # Compiles thermodynamic properties for each of the three cases into one array
    'Temperature (K)': temperatures,                         
    'U_Case1 (J)': U_case1,
    'F_Case1 (J)': F_case1,
    'S_Case1 (J/K)': S_case1,
    'U_Case2 (J)': U_case2,
    'F_Case2 (J)': F_case2,
    'S_Case2 (J/K)': S_case2,
    'U_Case3 (J)': U_case3,
    'F_Case3 (J)': F_case3,
    'S_Case3 (J/K)': S_case3
}
df = pd.DataFrame(data)                                        # Creates a Pandas dataframe

df.to_csv('thermodynamic_properties.csv', index = False)       # Saves thermodynamic properties dataframe as csv file

plt.figure(figsize = (12, 10))                                 # Defines plot size as 12 inches by 10 inches

plt.subplot(2, 2, 1)                                           # Creates a subplot
plt.plot(temperatures, Z_case1, label='Case 1 (Isolated)')     # Creates label for isolated Ce3+ ion
plt.plot(temperatures, Z_case2, label='Case 2 (SOC)')          # Creates label for Ce3+ ion with SOC
plt.plot(temperatures, Z_case3, label='Case 3 (SOC + CFS)')    # Creates label for Ce3+ ion with SOC and CSF
plt.xlabel('Temperature (K)')                                  # Creates x-axis label
plt.ylabel('Partition Function (Z)')                           # Creates y-axis label
plt.title('Partition Function vs Temperature')                 # Creates title
plt.legend()                                                   # Shows legend
plt.grid(True)                                                 # Shows gridlines

plt.subplot(2, 2, 2)                                           # Creates a subplot
plt.plot(temperatures, U_case1, label='Case 1 (Isolated)')     # Creates label for isolated Ce3+ ion
plt.plot(temperatures, U_case2, label='Case 2 (SOC)')          # Creates label for Ce3+ ion with SOC
plt.plot(temperatures, U_case3, label='Case 3 (SOC + CFS)')    # Creates label for Ce3+ ion with SOC and CSF
plt.xlabel('Temperature (K)')                                  # Creates x-axis label
plt.ylabel('Internal Energy (J)')                              # Creates y-axis label
plt.title('Internal Energy vs Temperature')                    # Creates title
plt.legend()                                                   # Shows legend
plt.grid(True)                                                 # Shows gridlines

plt.subplot(2, 2, 3)                                           # Creates a subplot
plt.plot(temperatures, F_case1, label='Case 1 (Isolated)')     # Creates label for isolated Ce3+ ion
plt.plot(temperatures, F_case2, label='Case 2 (SOC)')          # Creates label for Ce3+ ion with SOC
plt.plot(temperatures, F_case3, label='Case 3 (SOC + CFS)')    # Creates label for Ce3+ ion with SOC and CSF
plt.xlabel('Temperature (K)')                                  # Creates x-axis label
plt.ylabel('Free Energy (J)')                                  # Creates y-axis label
plt.title('Free Energy vs Temperature')                        # Creates title
plt.legend()                                                   # Shows legend
plt.grid(True)                                                 # Shows gridlinesplt.grid(True)

plt.subplot(2, 2, 4)                                           # Creates a subplot
plt.plot(temperatures, S_case1, label='Case 1 (Isolated)')     # Creates label for isolated Ce3+ ion
plt.plot(temperatures, S_case2, label='Case 2 (SOC)')          # Creates label for Ce3+ ion with SOC
plt.plot(temperatures, S_case3, label='Case 3 (SOC + CFS)')    # Creates label for Ce3+ ion with SOC and CSF
plt.xlabel('Temperature (K)')                                  # Creates x-axis label
plt.ylabel('Entropy (J/K)')                                    # Creates y-axis label
plt.title('Entropy vs Temperature')                            # Creates title
plt.legend()                                                   # Shows legend
plt.grid(True)                                                 # Shows gridlines

plt.tight_layout(pad=3.0)                                      # Adds additional padding to plot to avoid text overlap
plt.show()                                                     # Shows the plot

plt.savefig('thermodynamic_properties_plots.png')              # Saves the plot
