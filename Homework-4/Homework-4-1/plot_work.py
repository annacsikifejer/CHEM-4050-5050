import numpy as np                                                    # Imports the NumPy library and designates as np
import pandas as pd                                                   # Imports the Pandas library and designates as pd
from scipy.integrate import trapezoid                                 # Imports the trapezoid function from the SciPy integrate submodule of the SciPy library
import matplotlib.pyplot as plt                                       # Imports the pyplot submodule of the matplotlib module and designates as plt

def compute_work_isothermal(n, R, T, Vi, Vf, num_points = 100):       # Defines the isothermal work function

  V = np.linspace(Vi, Vf, num_points)                                 # Defines Volume as a NumPy array containing equally spaced values from initial volume (Vi) to final volume (Vf)
  P = (n * R * T) / V                                                 # Defines Pressure according to the ideal gas law (n = number of moles, R = ideal gas constant, T = temperature)
  work = -trapezoid(P, V)                                             # Defines work as the negative area found by trapezoids under the P-V curve
  return work                                                         # Returns the work for an isothermal process using the trapezoidal rule



def compute_work_adiabatic(P_i, V_i, V_f, gamma, num_points = 100):   # Defines the adiabatic work function using 100 equally space data points

  V = np.linspace(Vi, Vf, num_points)                                 # Defines Volume as a NumPy array containing 100 equally spaced values from initial Volume (Vi) to final Volume (Vf)
  constant = P_i * (V_i ** gamma)                                     # Defines the product of the initial Pressure (Pi), initial Volume (Vi), and adiabatic index (gamma) as a constant
  P = constant / (V ** gamma)                                         # Defines the Pressure (P) as a constant divded by the product of Volume (V) and the adiabatic index (gamma)
  work = -trapezoid(P, V)                                             # Defines work as the negative area found by trapezoids under the P-V curve
  return work                                                         # Returns the work for an isothermal process using the trapezoidal rule



n = 1                                                                 # Defines the number of moles (mol) as 1
R = 8.314                                                             # Defines the ideal gas constant (J/mol-K) as 8.314
T = 300                                                               # Defines the Temperature (K) as 300
Vi = 0.1                                                              # Defines the initial Volume (m^3) as 0.1
gamma = 1.4                                                           # Defines the adiabatic index (unitless) as 1.4                                                        
Vf_values = np.linspace(Vi, 3 * Vi, 100)                              # Defines the range of final Volumes (Vf) as a NumPy array of 100 equally spaced data points from the initial Volume (Vi) to three multiplied by the initial Volume (3 * Vi)

work_isothermal = [compute_work_isothermal(n, R, T, Vi, Vf, num_points = 100) for Vf in Vf_values]    # Defines the isothermal work over the range of final Volumes

P_i = (n * R * T) / Vi      # Defines the initial Pressure (P_i) for the adiabatic process as the product of the number of moles, ideal gas constant, and temperature, divided by the initial Volume (Vi)

work_adiabatic = [compute_work_adiabatic(P_i, Vi, Vf, gamma, num_points = 100) for Vf in Vf_values]   # Defines the adiabatic work over the range of final Volumes



plt.figure(figsize = (10, 6))                                                      # Defines plot size as 10 inches by 6 inches
plt.plot(Vf_values, work_isothermal, label = 'Isothermal Expansion')               # Plots the work done through isothermal expansion
plt.plot(Vf_values, work_adiabatic, label = 'Adiabatic Expansion')                 # Plots the work done through adiabatic expansion
plt.xlabel('Final Volume (m$^3$)')                                                 # Creates label for x-axis
plt.ylabel('Work Done (J)')                                                        # Creates label for y-axis
plt.title('Work Done During Isothermal and Adiabatic Expansion')                   # Creates title for plot
plt.legend()                                                                       # Shows legend
plt.grid(True)                                                                     # Shows gridlines
plt.show()                                                                         # Shows plot



results_df = pd.DataFrame({                                                        # Creates a dataframe storing the Volume and work results
    'Final Volume (m^3)': Vf_values,                                               # Stores the final Volume values in m^3
    'Work Isothermal (J)': work_isothermal,                                        # Stores the isothermal work values in J
    'Work Adiabatic (J)': work_adiabatic                                           # Stores the adiabatic work values in J
})

results_df.to_csv('work_vs_final_volume.csv', index = False)                       # Saves the dataframe as a csv file
print("CSV file 'work_vs_final_volume.csv' created successfully.")                 # Prints success message
