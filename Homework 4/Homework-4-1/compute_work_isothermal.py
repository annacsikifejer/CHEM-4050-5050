import numpy as np                                               # Imports the NumPy library and designates as np
from scipy.integrate import trapezoid                            # Imports the trapezoid function from the SciPy integrate submodule of the SciPy library

def compute_work_isothermal(n, R, T, Vi, Vf, num_points = 100):  # Defines the isothermal work function using 100 equally space data points

  V = np.linspace(Vi, Vf, num_points)                            # Defines Volume as a NumPy array containing 100 equally spaced values from initial Volume (Vi) to final Volume (Vf)
  P = (n * R * T) / V                                            # Defines Pressure according to the ideal gas law (n = number of moles, R = ideal gas constant, T = temperature)
  work = -trapezoid(P, V)                                        # Defines work as the negative area found by trapezoids under the P-V curve
  return work                                                    # Returns the work for an isothermal process using the trapezoidal rule
