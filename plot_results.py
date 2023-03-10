import os
import pandas as pd
import matplotlib.pyplot as plt
import re

# Define a regex pattern to extract the algorithm and min support from the file name
pattern = r'^(?!arules)(.+?)_(\d+)_([\d\.]+)$' 

# Create an empty DataFrame to store the execution times
df = pd.DataFrame(columns=['Algorithm', 'MinSupport', 'ExecutionTime'])

# Loop over all the files in the folder
folder_path = 'output'
for filename in os.listdir(folder_path):
    # Filter the files based on their name prefix using the isfile function
    if os.path.isfile(os.path.join(folder_path, filename)):
        # Extract the algorithm and min support from the file name using regex
        match = re.match(pattern, filename)
        if match:
            algorithm = match.group(1)
            min_support = float(match.group(3))

            # Read the last line of the file and extract the execution time
            with open(os.path.join(folder_path, filename)) as f:
                last_line = f.readlines()[-1]
                execution_time = float(last_line.split()[-2])

            # Append the data to the DataFrame
            df = df.append({'Algorithm': algorithm, 'MinSupport': min_support, 'ExecutionTime': execution_time},
                           ignore_index=True)

# Plot the scatter plot using matplotlib
fig, ax = plt.subplots()
for algorithm, data in df.groupby('Algorithm'):
    ax.scatter(data['MinSupport'], data['ExecutionTime'], label=algorithm)
ax.set_xlabel('Min Support')
ax.set_ylabel('Execution Time (seconds)')
ax.legend()
plt.show()
