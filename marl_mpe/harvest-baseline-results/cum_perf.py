import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the folders with CSV files
base_dir = '/home/angelsylvester/Documents/dynamic-rl/marl_mpe/harvest-baseline-results'

# Initialize data dictionary to store performance data for each directory
data = {}

# Iterate through directories
for folder in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder)
    if os.path.isdir(folder_path):
        csv_file = os.path.join(folder_path, 'csv_cum.csv')
        if os.path.isfile(csv_file):
            # Read CSV file
            df = pd.read_csv(csv_file)

            # Store performance data in data dictionary
            data[folder] = df[['Iteration', 'Average_Reward']]

# Plotting the line chart for each directory's performance
plt.figure(figsize=(10, 6))
for folder, df in data.items():
    plt.plot(df['Iteration'], df['Average_Reward'], label=folder)

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Average Reward Over Iterations for Each Directory')
plt.legend()
plt.grid(True)
plt.show()
