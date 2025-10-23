import numpy as np                    # Imports the NumPy library as np

np.random.seed(42)                    # Sets a random seed



def hydrogen_1s_orbital(point):                                                               # Defines the normalized H 1s orbital
    a0 = 1.0                                                                                  # Sets the Bohr radius equal to 1.0 Angstroms
    r = np.linalg.norm(point)                                                                 # Defines the magnitude of the radial distance
    return (1.0 / np.sqrt(np.pi * a0**3)) * np.exp(-r / a0)                                   # Returns the normalized H 1s orbital

def hydrogen_1s_laplacian(point):                                                             # Defines the laplacian of the H 1s orbital
    a0 = 1.0                                                                                  # Sets the Bohr radius equal to 1.0 Angstroms
    r = np.linalg.norm(point)                                                                 # Defines the magnitude of the radial distance
    return (1.0 / np.sqrt(np.pi * a0**3)) * (1.0 / a0**2) * (r / a0 - 2) * np.exp(-r / a0)    # Returns the laplacian of the H 1s orbital



def random_sampling_monte_carlo(num_points):    # Defines the integral for a sampling region
    box_half_length = 5.0                       # Defines half-length of box as sampling region
    volume = (2 * box_half_length)**3           # Defines box volume as full box length cubed

    points = np.random.uniform(-box_half_length, box_half_length, size = (num_points, 3))                                   # Generates random points within box volume

    integrand_values = np.array([-0.5 * hydrogen_1s_orbital(point) * hydrogen_1s_laplacian(point) for point in points])     # Calculates the integrand at each point in range

    estimated_integral = np.mean(integrand_values) * volume                     # Calculates the estimated integral

    return estimated_integral                                                   # Returns the estimated integral (ie diagonal kinetic energy matrix element)

num_points_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]      # Defines the number of points of interest (10^2 to 10^8)
results_random_sampling = {}                                                    # Creates a dictionary to store number of points

for num_points in num_points_list:                                              # Looks at each number of sampling points in the dictionary
    estimated_value = random_sampling_monte_carlo(num_points)                   # Calculates diagonal kinetic energy matrix element using Monte Carlo with random sampling
    results_random_sampling[num_points] = estimated_value                       # Stores diagonal kinetic energy matrix element as the estimated value
    print(f"Estimated value at {num_points} points = {estimated_value:.3f}")    # Prints diagonal kinetic energy matrix element to three decimal places



def importance_sampling_monte_carlo(num_points):                         # Defines the estimated orbital integral from importance sampling
    mean = np.array([0.0, 0.0, 0.0])                                     # Creates an array in 3d space (x,y,z)
    std_dev = 1.0                                                        # Sets the standard deviation at 1.0

    points = np.random.normal(mean, std_dev, size = (num_points, 3))     # Generates random points from the distribution

    r_squared = np.sum(points**2, axis = 1)                                                                      # Calculates the distance of a point in 3D space
    gaussian_pdf_values = (1.0 / (std_dev * np.sqrt(2 * np.pi))**3) * np.exp(-r_squared / (2 * std_dev**2))      # Calculates a probability density function
    
    original_integrand_values = np.array([-0.5 * hydrogen_1s_orbital(point) * hydrogen_1s_laplacian(point) for point in points])      # Creates an array of integrand values

    epsilon = 1e-10                                                                                              # Adds a small epsilon value to avoid dividing by zero
    importance_sampling_integrand_values = original_integrand_values / (gaussian_pdf_values + epsilon)           # Calculates the integrand values using importance sampling

    estimated_integral = np.mean(importance_sampling_integrand_values)                                           # Calculates the estimated integral from the average of all values

    return estimated_integral                                                                                    # Returns the estimated diagonal kinetic energy matrix element

num_points_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]                                       # Creates a list of each number of points (10^2 to 10^8)
results_importance_sampling = {}                                                                                 # Creates a dictionary of the diagonal kinetic energy matrix element values

for num_points in num_points_list:                                                                               # Looks at each number from sampling numbers list
    estimated_value = importance_sampling_monte_carlo(num_points)                                                # Calculates the estimated diagonal kinetic energy matrix element
    results_importance_sampling[num_points] = estimated_value                                                    # Stores the element value as an estimated value
    print(f"Number of points: {num_points}, Estimated value (Importance Sampling): {estimated_value:.3f}")       # Prints diagonal kinetic energy matrix element with importance sampling to 3 decimal places



def hydrogen_1s_orbital_shifted(point, shift):                                  # Defines the off-diagonal H 1s orbital
    shifted_point = point - shift                                               # Shifts orbital to be off-diagonal
    return hydrogen_1s_orbital(shifted_point)                                   # Returns the off-diagonal H 1s orbital

def hydrogen_1s_laplacian_shifted(point, shift):                                # Defines the off-diagonal H 1s laplacian
    shifted_point = point - shift                                               # Shifts orbital to be off-diagonal
    return hydrogen_1s_laplacian(shifted_point)                                 # Returns the off-diagonal H 1s laplacian

def random_sampling_off_diagonal(num_points):                                   # Defines the off-diagonal sampling region
    box_half_length = 5.0                                                       # Defines half-length of box as sampling region
    volume = (2 * box_half_length)**3                                           # Defines box volume as full box length cubed
    shift_vector = np.array([1.0, 0.0, 0.0])                                    # Shifts the box to be off-diagonal (x = 1.0 Angstroms)
    integrand_values = np.array([-0.5 * hydrogen_1s_orbital(point) * hydrogen_1s_laplacian_shifted(point, shift_vector) for point in points])     # Calculates the integrand at each point in range
    points = np.random.uniform(-box_half_length, box_half_length, size = (num_points, 3))                                                         # Generates random points within box volume

    estimated_integral = np.mean(integrand_values) * volume                     # Calculates the estimated integral

    return estimated_integral                                                   # Returns the estimated integral (ie off-diagonal kinetic energy matrix element)

num_points_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]                                            # Creates a list of each number of points (10^2 to 10^8)
results_importance_sampling = {}                                                                                      # Creates a dictionary of the diagonal kinetic energy matrix element values

for num_points in num_points_list:                                                                                    # Looks at each number of sampling points in the dictionary
    estimated_value = random_sampling_off_diagonal(num_points)                                                        # Calculates off-diagonal kinetic energy matrix element using random sampling
    results_off_diagonal_random_sampling[num_points] = estimated_value                                                # Stores off-diagonal kinetic energy matrix element as the estimated value
    print(f"Number of points: {num_points}, Estimated off-diagonal value (Random Sampling): {estimated_value:.3f}")   # Prints off-diagonal element values with random sampling



def importance_sampling_monte_carlo(num_points):                         # Defines the estimated orbital integral from importance sampling
    mean = np.array([0.0, 0.0, 0.0])                                     # Creates an array in 3d space (x,y,z)
    std_dev = 1.0                                                        # Sets the standard deviation at 1.0

    points = np.random.normal(mean, std_dev, size = (num_points, 3))     # Generates random points from the distribution

    r_squared = np.sum(points**2, axis = 1)                                                                      # Calculates the distance of a point in 3D space
    gaussian_pdf_values = (1.0 / (std_dev * np.sqrt(2 * np.pi))**3) * np.exp(-r_squared / (2 * std_dev**2))      # Calculates a probability density function
    
    original_integrand_values = np.array([-0.5 * hydrogen_1s_orbital(point) * hydrogen_1s_laplacian(point) for point in points])      # Creates an array of integrand values

    epsilon = 1e-10                                                                                              # Adds a small epsilon value to avoid dividing by zero
    importance_sampling_integrand_values = original_integrand_values / (gaussian_pdf_values + epsilon)           # Calculates the integrand values using importance sampling

    estimated_integral = np.mean(importance_sampling_integrand_values)                                           # Calculates the estimated integral from the average of all values

    return estimated_integral                                                                                    # Returns the estimated diagonal kinetic energy matrix element


num_points_list = [100, 1000, 10000, 100000, 1000000, 10000000, 100000000]                                       # Creates a list of each number of points (10^2 to 10^8)
results_importance_sampling = {}                                                                                 # Creates a dictionary of the off-diagonal kinetic energy matrix element values

for num_points in num_points_list:                                                                               # Looks at each number of sampling points in the dictionary
    estimated_value = importance_sampling_monte_carlo(num_points)                                                # Calculates off-diagonal kinetic energy matrix element using importance sampling
    results_importance_sampling[num_points] = estimated_value                                                    # Stores the element value as an estimated value
    print(f"Number of points: {num_points}, Estimated value (Importance Sampling): {estimated_value:.3f}")       # Prints the off-diagonal kinetic energy matrix element values to 3 decimal places
