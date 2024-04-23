import os
import pandas as pd
import matplotlib.pyplot as plt

# Directory containing the folders with CSV files
base_dir = '/home/angelsylvester/Documents/dynamic-rl/marl_mpe/clean-baseline-results'

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

            # For 'joint' folder, divide Average_Reward by number of agents (5 in this case)
            if folder == 'joint':
                df['Average_Reward'] /= 5  # Divide by number of agents

            # Store performance data in data dictionary
            data[folder] = df[['Iteration', 'Average_Reward']]

# Plotting the line chart for each directory's performance
plt.figure(figsize=(10, 6))
for folder, df in data.items():
    plt.plot(df['Iteration'], df['Average_Reward'], label=folder)

plt.xlabel('Iteration')
plt.ylabel('Average Reward')
plt.title('Average Reward for Clean-up Env')
plt.legend()
plt.grid(True)
plt.show()
