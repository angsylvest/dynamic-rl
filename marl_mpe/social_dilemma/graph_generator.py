import pandas as pd
import matplotlib.pyplot as plt
import os

import ast 

# Assuming your data is saved in a file named 'data.csv'
avg_directory = ""
individual_perf = "" 
custom_metrics = ""

''' --------- Avg Performance Plot -------------'''
if avg_directory != "": 
    # Create a line plot for each file
    for filename in os.listdir(avg_directory):
        if filename.endswith('.csv'):  # Assuming your files have a .csv extension
            filepath = os.path.join(avg_directory, filename)
            file_data = pd.read_csv(filepath)
            plt.plot(file_data['Iteration'], file_data['Average_Reward'], label=filename)

    # Set plot labels and legend
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward vs. Iteration')

    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('line_plot.png')  # Change the filename and extension as needed

    # Show the plot
    plt.show()


'''--------- Individual Performance Plot --------- '''

if individual_perf != "": 
    # Load the data into a DataFrame
    df = pd.read_csv(individual_perf)

    # Create a line plot for each Agent_ID
    for agent_id, agent_data in df.groupby('Agent_ID'):
        plt.plot(agent_data['Iteration'], agent_data['Average_Reward'], label=f'Agent {agent_id}')

    # Set plot labels and legend
    plt.xlabel('Iteration')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward vs. Iteration for Each Agent')

    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('line_plot.png')  # Change the filename and extension as needed

    # Show the plot
    plt.show()


'''--------- Custom Metrics Plot --------- '''

if custom_metrics != "": 
    # Initialize a dictionary to store average num_collected per iteration for each file
    avg_num_collected = {}

    # Iterate through each CSV file in the directory
    for filename in os.listdir(custom_metrics):
        if filename.endswith('.csv'):
            filepath = os.path.join(custom_metrics, filename)
            # Read the CSV file
            df = pd.read_csv(filepath)
            # Parse the 'Custom Metrics' column
            df['Custom Metrics'] = df['Custom Metrics'].apply(ast.literal_eval)
            # Extract 'num_collected' values for each agent
            for iteration, metrics in df.groupby('Iteration')['Custom Metrics']:
                for agent, data in metrics.items():
                    avg_num_collected.setdefault(agent, []).append(data['num_collected'])

    # Calculate the average num_collected per iteration for each agent
    for agent, values in avg_num_collected.items():
        avg_num_collected[agent] = [sum(x) / len(x) for x in zip(*values)]

    # Plot the average num_collected per iteration for each agent
    for agent, values in avg_num_collected.items():
        plt.plot(range(1, len(values) + 1), values, label=f'Agent {agent}')

    # Set plot labels and legend
    plt.xlabel('Iteration')
    plt.ylabel('Average Num Collected')
    plt.legend()
    plt.title('Average Num Collected per Iteration')

    # Save the plot as an image file (e.g., PNG or PDF)
    plt.savefig('line_plot.png')  # Change the filename and extension as needed

    # Show the plot
    plt.show()