import pandas as pd                            # Imports the Pandas library and saves as pd
import numpy as np                             # Imports the NumPy library and saves as np
import scipy.optimize                          # Imports the Optimize module of the SciPy library
from scipy.optimize import minimize            # Imports the Minimize submodule of the Optimize module
import matplotlib.pyplot as plt                # Imports the Pyplot submodule of the MatPlot Library and saves as plt
import statsmodels.api as sm                   # Imports the statsmodels api and saves as sm
import os                                      # Imports the os module to save plot as png


df = pd.read_csv('trouton.csv')                # Reads the trouton csv file and creates a dataframe

def objective_function(params, T_B, H_v):      # Defines the objective function of the Least Squares Error
    a, b = params                              # Defines parameters a and b
    predicted_H_v = a * T_B + b
    return np.sum((H_v - predicted_H_v)**2)

initial_guess = [0.1, 10]                      # Initial guess for optimization

result = minimize(objective_function, initial_guess, args=(df['T_B (K)'], df['H_v (kcal/mol)']))    # Uses minimize to find optimal a and b parameters

optimal_a, optimal_b = result.x                                                                     # Stores optimal a and b parameters

df['Hv_J_mol_K'] = df['H_v (kcal/mol)'] * 4184 / df['T_B (K)']                                      # Converts units to J / mol*K


plt.figure(figsize = (10, 6))                          # Defines figure size

if 'Class' in df.columns:                              # Makes each class into a distinct color
    classes = df['Class'].unique()
    colors = ['orange', 'yellow', 'red', 'lightgreen']
    for i, class_name in enumerate(classes):
        class_df = df[df['Class'] == class_name]
        plt.scatter(class_df['T_B (K)'], class_df['Hv_J_mol_K'], label=class_name, color=colors[i % len(colors)])

T_B_fit = np.linspace(df['T_B (K)'].min(), df['T_B (K)'].max(), 100)
Hv_fit = optimal_a * T_B_fit + optimal_b
plt.plot(T_B_fit, Hv_fit, color='grey', label=f'Fit: Hv = {optimal_a:.2f} * TB + {optimal_b:.2f}')

equation_text = f'Hv = {optimal_a:.2f} * TB + {optimal_b:.2f}'                                       # Writes optimized equation
plt.text(0.05, 0.95, equation_text, transform = plt.gca().transAxes, fontsize = 10, verticalalignment = 'top', bbox = dict(boxstyle = 'round,pad = 0.5', fc = 'wheat', alpha = 0.5))
# Creates a text box on plot to display optimized equation


plt.xlabel('TB (K)')                                   # Creates x-axis label
plt.ylabel('Hv (J/mol-K)')                             # Creates y-axis label
plt.title('Trouton’s Rule Optimization')               # Creates title
plt.legend()                                           # Creates legend
plt.grid(True)                                         # Turns on Gridlines

output_folder = 'Homework-3-2'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

print("Result Comparison: the slope of the optimization is considerably lower than that of the linear regression performed in Homework-3-1. ")
print("Data Interpretation: In the case of this problem, a linear regression is a more appropriate method, as the number of data points above and below the least squares regression line were roughly equal. In contrast, all data points lie above the Trouton's Rule optimization line, indicating a poor fit to the data and that compounds with the same properties do not follow the same linear relationship as other compounds. As a result, Trouton's Rule may be appropriate for models of compounds with similar chemical interactions, ie one of the four classes rather than all at once") 

plt.savefig(os.path.join(output_folder, 'troutons_rule_optimization.png'))
