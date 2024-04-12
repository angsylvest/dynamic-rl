import os
import pandas as pd
import matplotlib.pyplot as plt

show_avg = True 
show_diff = True 

# Directory containing the folders with CSV files
base_dir = '/home/angelsylvester/Documents/dynamic-rl/marl_mpe/harvest-baseline-results'

if show_avg: 
    # Initialize data dictionary to store averaged values
    data = {}

    # Iterate through folders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, 'csv_perf.csv')
            if os.path.isfile(csv_file):
                # Read CSV file
                df = pd.read_csv(csv_file)

                # Extract 'num_collected' values for each agent
                df['Custom Metrics'] = df['Custom Metrics'].apply(eval)  # Convert string to dictionary
                for agent_id in df['Custom Metrics'][0].keys():
                    df[agent_id] = df['Custom Metrics'].apply(lambda x: x[agent_id]['num_collected'])

                # Calculate average 'num_collected' across all agents
                df['Average Collected'] = df.drop(['Iteration', 'Custom Metrics'], axis=1).mean(axis=1)

                # Store averaged values in data dictionary
                data[folder] = df[['Iteration', 'Average Collected']]

    # Plotting the line chart
    plt.figure(figsize=(10, 6))
    for folder, df in data.items():
        plt.plot(df['Iteration'], df['Average Collected'], label=folder)

    plt.xlabel('Iteration')
    plt.ylabel('Average Number of Collected Items')
    plt.title('Average Number of Collected Items for Harvest Env')
    plt.legend()
    plt.grid(True)
    plt.show()


if show_diff: 
    # Initialize data dictionary to store differences
    data = {}

    # Iterate through folders
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, 'csv_perf.csv')
            if os.path.isfile(csv_file):
                # Read CSV file
                df = pd.read_csv(csv_file)

                # Convert 'Custom Metrics' to dictionary
                df['Custom Metrics'] = df['Custom Metrics'].apply(eval)

                # Extract 'num_collected' values for each agent
                for agent_id in df['Custom Metrics'][0].keys():
                    df[agent_id] = df['Custom Metrics'].apply(lambda x: x[agent_id]['num_collected'])

                # Calculate difference between max and min 'num_collected' for each agent
                df['Difference'] = df.drop(['Iteration', 'Custom Metrics'], axis=1).apply(lambda row: row.max() - row.min(), axis=1)

                # Store differences in data dictionary
                data[folder] = df[['Iteration', 'Difference']]

    # Plotting the difference for each agent at each iteration
    plt.figure(figsize=(10, 6))
    for folder, df_diff in data.items():
        for column in df_diff.columns[1:]:  # Exclude 'Iteration' column
            plt.plot(df_diff['Iteration'], df_diff[column], label=f'{folder} - {column}')

    plt.xlabel('Iteration')
    plt.ylabel('Difference (Max - Min) Collected Items')
    plt.title('Difference Between Max and Min Collected Items by Agent Over Iterations')
    plt.legend()
    plt.grid(True)
    plt.show()