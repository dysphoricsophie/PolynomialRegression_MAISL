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
    range_data, altitude_data = [], []
    with open(file_path, "r") as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            range_data.append(float(row[1]))
            altitude_data.append(float(row[4]))
    return np.array(range_data), np.array(altitude_data)

def csv_edit(al):
    fields = ['Launch Angle', 'Best Fitting Polynomial Order']
    rows = al
    filename = "polynomial_recommendation.csv"

    with open(filename, 'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(fields)
        csvwriter.writerows(rows)

rmse_arr = []; r2_arr = []
rmse_for_all = []
r2_for_all = []

rmseA3 = []; r2A3 = []
rmseA4 = []; r2A4 = []
rmseA5 = []; r2A5 = []
rmseA6 = []; r2A6 = []
rmseA7 = []; r2A7 = []
rmseA8 = []; r2A8 = []

for j in range(3, 9):
    # Define output directory paths with better variable names
    output_dir = "Output_Files"
    altitude_range_dir = os.path.join(output_dir, f"Altitude_Range{j}")

    # Create directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(altitude_range_dir, exist_ok=True)
    with open(f"Output_Files\Altitude_Range{j}\eqtns.txt", "x") as file:
        file.write("")

    # Set polynomial degree
    poly_degree = j

    rmse_arre = []
    r2_arre = []
    for simulation_number in range(1, 61):
        # Construct file paths with f-strings
        data_file = f"Data Save File\Simulation{simulation_number} - Data Points.csv"
        range_data, altitude_data = read_csv_data(data_file)

        # Reshape data for compatibility with sklearn (single feature needs reshaping)
        altitude_data = altitude_data.reshape(-1, 1)  # Equivalent to x[:, np.newaxis]

        # Create polynomial features
        polynomial_converter = PolynomialFeatures(degree=poly_degree)
        x_poly = polynomial_converter.fit_transform(altitude_data)

        # Train linear regression model
        model = LinearRegression()
        model.fit(x_poly, range_data)
        predicted_range = model.predict(x_poly)

        # Calculate root mean squared error (RMSE)
        rmse = np.sqrt(mean_squared_error(range_data, predicted_range))
        rmse_arre.append(rmse)

        # Calculate R-squared score
        r2 = r2_score(range_data, predicted_range)
        r2_arre.append(r2)

        # Sort data by altitude for plotting
        sorted_data = sorted(zip(altitude_data.ravel(), predicted_range.ravel()))
        sorted_altitude, sorted_predicted_range = zip(*sorted_data)

        # Create the plot
        plt.figure(figsize=(12, 6))
        plt.title('Rocket Trajectory - Altitude vs. Range')
        plt.xlabel('Range (m)')
        plt.ylabel('Altitude (m)')
        plt.plot(sorted_altitude, sorted_predicted_range, color='m', label=f'Fitted Polynomial (R² = {r2:.3f})')
        plt.scatter(altitude_data.ravel(), range_data, s=20, label='Data Points')
        plt.xlim([sorted_altitude[0] - 1, sorted_altitude[-1] + 1])
        plt.legend()

        # Save plot with descriptive filename
        plot_filename = f"{altitude_range_dir}/fig{simulation_number}.png"
        plt.savefig(plot_filename)
        plt.close()

        with open(f"Output_Files\Altitude_Range{j}\eqtns.txt", "a") as file:
            file.write(f"The RMSE value of the fitted polynomial at launch angle {simulation_number} is: \n{rmse}\n")
            file.write(f"The R² value of the fitted polynomial at launch angle {simulation_number} is: \n{r2}\n\n")
            file.close()

    match j:
        case 3:
            for z in rmse_arre:
                rmseA3.append(z)
            for x in r2_arre:
                r2A3.append(x)
        case 4:
            for z in rmse_arre:
                rmseA4.append(z)
            for x in r2_arre:
                r2A4.append(x)
        case 5:
            for z in rmse_arre:
                rmseA5.append(z)
            for x in r2_arre:
                r2A5.append(x)
        case 6:
            for z in rmse_arre:
                rmseA6.append(z)
            for x in r2_arre:
                r2A6.append(x)
        case 7:
            for z in rmse_arre:
                rmseA7.append(z)
            for x in r2_arre:
                r2A7.append(x)
        case 8:
            for z in rmse_arre:
                rmseA8.append(z)
            for x in r2_arre:
                r2A8.append(x)
for k in range(0, 60):
    a = [rmseA3[k], rmseA4[k], rmseA5[k], rmseA6[k], rmseA7[k], rmseA8[k]]
    b = [r2A3[k], r2A4[k], r2A5[k], r2A6[k], r2A7[k], r2A8[k]]
    rmse_for_all.append(f"{min(a)} with a {round((a.index(min(a)))+4, 3)}th order")
    r2_for_all.append(f"{max(b)} with a {round((b.index(max(b)))+4, 3)}th order")

best_fit = []
with open(f"Output_Files\eval.txt", "x") as file:
    file.write("")
for h in range(len(rmse_for_all)):
    print(f"Best fitted polynomial regression order at a launch angle of {h+1} according to the RMSE value is {rmse_for_all[h]}")
    print(f"Best fitted polynomial regression order at a launch angle of {h+1} according to the R² value is {r2_for_all[h]}\n")
    with open(f"Output_Files\eval.txt", "a") as file:
        file.write(f"Best fitted polynomial regression order at a launch angle of {h+1} according to the RMSE value is {rmse_for_all[h]}\n")
        file.write(f"Best fitted polynomial regression order at a launch angle of {h+1} according to the R² value is {r2_for_all[h]}\n\n")
        file.close()
    best_fit.append(int((rmse_for_all[h].split(" with a ")[1]).replace("th order", "")))
laun_ang = np.linspace(1, 60, 60)
all_them = []
for c in range(60):
    all_them.append([int(laun_ang[c]), best_fit[c]])
csv_edit(all_them)
