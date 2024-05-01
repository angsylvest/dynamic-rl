import os
import pandas as pd
import matplotlib.pyplot as plt

show_avg = False 
show_diff = False 
show_stdev = False
show_clean_up = True 

# Directory containing the folders with CSV files
base_dir = '/home/angelsylvester/Documents/dynamic-rl/marl_mpe/clean-baseline-results'

if show_avg: 
    # Initialize data dictionary to store averaged values
    data = {}
    std_devs = {}

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
    plt.title('Average Number of Collected Items for Cleanup Env')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0, right=750)
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
    plt.xlim(left=0, right=750)
    plt.show()


if show_stdev:
    # Initialize data dictionary to store total resources collected
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

                # Calculate total resources collected for each iteration
                df['Total Collected'] = df.drop(['Iteration', 'Custom Metrics'], axis=1).sum(axis=1)

                # Store total resources collected in data dictionary
                data[folder] = df[['Iteration', 'Total Collected']]

    # Plotting the line chart
    plt.figure(figsize=(10, 6))
    for folder, df in data.items():
        plt.plot(df['Iteration'], df['Total Collected'], label=folder)

    plt.xlabel('Iteration')
    plt.ylabel('Total Resources Collected')
    plt.title('Total Resources Collected over Iterations for Harvest Env')
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0, right=750)
    plt.show()



# Initialize data dictionary to store distances from mean
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

            # Calculate the agent who collected the most for each iteration
            max_collected_agent = df.drop(['Iteration', 'Custom Metrics'], axis=1).idxmax(axis=1)
            max_collected_value = df.drop(['Iteration', 'Custom Metrics'], axis=1).max(axis=1)
            # Calculate the mean 'num_collected' across all agents for each iteration
            df['Mean Collected'] = df.drop(['Iteration', 'Custom Metrics'], axis=1).mean(axis=1)
            # Calculate the distance from mean for the agent who collected the most
            df['Distance from Mean'] = max_collected_value - df['Mean Collected']

            # Store distances from mean in data dictionary
            data[folder] = df[['Iteration', 'Distance from Mean']]

# Plotting the line chart
plt.figure(figsize=(10, 6))
for folder, df in data.items():
    plt.plot(df['Iteration'], df['Distance from Mean'], label=folder)

plt.xlabel('Iteration')
plt.ylabel('Distance from Mean for Max Collected Agent')
plt.title('Distance from Mean for Agent with Max Collected over Iterations for Harvest Env')
plt.legend()
plt.grid(True)
plt.xlim(left=0, right=750)
plt.show()

if show_clean_up:
    # Initialize data dictionary to store average number of 'num_cleaned' for each directory
    data = {}

    # Iterate through directories
    for folder in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder)
        if os.path.isdir(folder_path):
            csv_file = os.path.join(folder_path, 'csv_perf.csv')
            if os.path.isfile(csv_file):
                # Read CSV file
                df = pd.read_csv(csv_file)

                # Convert 'Custom Metrics' to dictionary
                df['Custom Metrics'] = df['Custom Metrics'].apply(eval)

                # Extract 'num_cleaned' values for each agent and calculate average
                df['Num Cleaned Avg'] = df['Custom Metrics'].apply(lambda x: sum(agent_data['num_cleaned'] for agent_data in x.values()) / len(x))

                # Store average number of 'num_cleaned' in data dictionary
                data[folder] = df[['Iteration', 'Num Cleaned Avg']]

    # Plotting the line chart for average number of 'num_cleaned' for each directory
    plt.figure(figsize=(10, 6))
    for folder, df in data.items():

        if folder == 'bayes':  # Check if the folder is Bayes
            plt.plot(df['Iteration'], df['Num Cleaned Avg'], label=folder, color='black')  # Set line color to black for Bayes folder
        else:
            plt.plot(df['Iteration'], df['Num Cleaned Avg'], label=folder)

    plt.xlabel('Iteration')
    plt.ylabel('Average Number of Cleaned Items')
    plt.title('Average Number of Cleaned Items')
    plt.xlim(left=0, right=750)
    plt.legend()
    plt.grid(True)
    plt.show()