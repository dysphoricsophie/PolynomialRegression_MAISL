import csv
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

def read_csv_data(file_path):
    """
    Reads data from a CSV file and returns separate NumPy arrays for time and altitude.
    """
    range_dataa, altitude_dataa = [], []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            range_dataa.append(float(row[1]))
            altitude_dataa.append(float(row[4]))
    return np.array(range_dataa), np.array(altitude_dataa)

# Define output directory paths with better variable names
output_dir = "Output_Files"
altitude_range_dir = os.path.join(output_dir, f"Altitude_Range")

# Create directories if they don't exist
os.makedirs(output_dir, exist_ok=True)
os.makedirs(altitude_range_dir, exist_ok=True)
with open(f"Output_Files\Altitude_Range\eqtns.txt", "x") as file:
    file.write("")

for simulation_number in range(1, 61):
    # Construct file paths with f-strings
    data_file = f"Data Save File\Simulation{simulation_number} - Data Points.csv"
    range_data, altitude_data = read_csv_data(data_file)

    # Reshape data for compatibility with sklearn (single feature needs reshaping)
    altitude_data = altitude_data.reshape(-1, 1)  # Equivalent to x[:, np.newaxis]

    poly_degree = None
    # Create polynomial features
    if 0 <= simulation_number <= 43:
        poly_degree = 4
    elif 43 <= simulation_number <= 56:
        poly_degree = 5
    elif simulation_number >= 57:
        poly_degree = 6

    polynomial_converter = PolynomialFeatures(degree=poly_degree)
    x_poly = polynomial_converter.fit_transform(altitude_data)

    # Train linear regression model
    model = LinearRegression()
    model.fit(x_poly, range_data)
    predicted_range = model.predict(x_poly)

    # Calculate root mean squared error (RMSE)
    rmse = np.sqrt(mean_squared_error(range_data, predicted_range))

    # Calculate R-squared score
    r2 = r2_score(range_data, predicted_range)

    # Sort data by altitude for plotting
    sorted_data = sorted(zip(altitude_data.ravel(), predicted_range.ravel()))
    sorted_altitude, sorted_predicted_range = zip(*sorted_data)

    # Create the plot
    plt.figure(figsize=(12, 6))
    plt.title(f'Rocket Trajectory Polynomial Regression - Altitude v/s Range Graph')
    plt.xlabel('Range (m)')
    plt.ylabel('Altitude (m)')
    plt.plot(sorted_altitude, sorted_predicted_range, color='m', label=f'{poly_degree}th degree Fitted Polynomial (R² = {r2:.3f})')
    plt.scatter(altitude_data.ravel(), range_data, s=20, label='Data Points')
    plt.xlim([sorted_altitude[0] - 1, sorted_altitude[-1] + 1])
    plt.legend()

    # Save plot with descriptive filename
    plot_filename = f"{altitude_range_dir}/fig{simulation_number}.png"
    plt.savefig(plot_filename)
    plt.close()

    with open(f"Output_Files\Altitude_Range\eqtns.txt", "a") as file:
        file.write(f"The RMSE value of the {poly_degree}th degree fitted polynomial at launch angle {simulation_number} is: \n{rmse}\n")
        file.write(f"The R² value of the {poly_degree}th degree fitted polynomial at launch angle {simulation_number} is: \n{r2}\n\n")
        file.close()