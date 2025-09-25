import pandas as pd                            # Imports the Pandas library and saves as pd
import numpy as np                             # Imports the NumPy library and saves as np
import scipy.optimize                          # Imports the Optimize module of the SciPy library
import matplotlib.pyplot as plt                # Imports the Pyplot submodule of the MatPlot Library and saves as plt
import statsmodels.api as sm                   # Imports the statsmodels api and saves as sm
import os                                      # Imports the os module to save plot as png


def ols_slope(x, y):                                     # Defines Ordinary Least Squares (OLS) line slope
    x_mean = np.mean(x)                                  # Defines mean of x value
    y_mean = np.mean(y)                                  # Defines mean of y value
    numerator = np.sum((x - x_mean) * (y - y_mean))      # Defines numerator of slope equation
    denominator = np.sum((x - x_mean) ** 2)              # Defines denominator of slope equation
    return numerator / denominator                       # Returns value of OLS slope (numerator divided by denominator)

def ols_intercept(x, y):                                 # Defines Ordinary Least Squares (OLS) intercept
    slope = ols_slope(x, y)
    return y_mean - slope * x_mean


df = pd.read_csv('trouton.csv')                                          # Reads the trouton csv file and creates a dataframe

df['Enthalpy of Vaporization (kJ/mol)'] = df['H_v (kcal/mol)'] * 4.184   # Converts units to kJ / mol

X = df['T_B (K)']                                                        # Creates a dataframe of x values (temperature)
X = sm.add_constant(X)                                                   # Adds a constant term for linear regression (ie the b term in y = mx + b)
y = df['Enthalpy of Vaporization (kJ/mol)']                              # Creates a dataframe of y values (enthalpy of vaporization)

model = sm.OLS(y, X).fit()

slope = model.params.iloc[1]
intercept = model.params.iloc[0]
conf_int = model.conf_int(alpha=0.05)
intercept_ci = conf_int.loc['const']
slope_ci = conf_int.loc['T_B (K)']

slope_j_mol_k = slope * 1000                               # Converts units to J / mol*K


plt.figure(figsize = (10, 6))                              # Defines size of plot

if 'Class' in df.columns:                                  # Assigns each class in dataframe a color
    classes = df['Class'].unique()                         # Determines unique classes
    colors = ['orange', 'yellow', 'red', 'lightgreen']     # Assigns colors from four options: orange, yellow, red, and light green
    for i, class_name in enumerate(classes):               # Assigns 1 color for each class
        class_df = df[df['Class'] == class_name]           # Creates a class dataframe based on classes in data
        plt.scatter(class_df['T_B (K)'], class_df['Enthalpy of Vaporization (kJ/mol)'], label = class_name, color = colors[i % len(colors)])      # Creates a scatter plot of data

plt.plot(df['T_B (K)'], model.predict(X), color = 'grey', label = 'Fitted Line')      # Plots the linear fit line in grey         

equation_text = f'Hv = {slope_j_mol_k:.2f} (±{ (slope_ci[1] - slope_ci[0]) / 2 * 1000:.2f}) J/mol-K * TB + {intercept:.2f} (±{(intercept_ci[1] - intercept_ci[0])/2:.2f}) kJ/mol'
# Creates linear fit equation, rounded to 2 decimal places

plt.text(0.05, 0.95, equation_text, transform = plt.gca().transAxes, fontsize = 10, verticalalignment='top', bbox=dict(boxstyle = 'round,pad = 0.5', fc = 'wheat', alpha = 0.5))
# Creates easy-to-read text box on plot to display linear fit equation

plt.xlabel(r'T(B) (K)')                                               # Creates a label for the x-axis
plt.ylabel(r'H(V) (kJ/mol)')                                          # Creates a label for the y-axis
plt.title('Trouton’s Rule')                                           # Creates a label for the title
plt.legend()                                                          # Creates a legend
plt.grid(True)                                                        # Turns on gridlines


output_dir = 'Homework-3-1'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print("Slope Interpretation: The approximation of the entropy of vaporization using least squares regression is 103.85 J / mol*K")

print("Trouton's Rule Comparison: Compared to the expected value for entropy of vaporization for many substances according to Trouton's Rule, 88 J / mol*K, the data collected show an overall higher value for the entropy of vaporization. This may be due to interactions between molecules, particuarly for metals and imperfect liquids")

print("Uncertainty: The uncertainty of slope a with 95% confidence is +/- 6.40 J / mol*K, while the uncertainty of y-intercept b with 95% confidence is +/- 6.21 kJ / mol.") 

plt.savefig(os.path.join(output_dir, 'troutons_rule_plot.png'))       # Saves plot as png in Homework-3-1
