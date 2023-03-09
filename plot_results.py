import matplotlib.pyplot as plt
import re
import os
import numpy as np
import pandas as pd

# Define a function to extract min_support from the output filename
def extract_min_support(filename):
    pattern = r"_([0-9]+\.[0-9]+)$"
    match = re.search(pattern, filename)
    if match:
        return float(match.group(1))
    else:
        return None

def plot_execution_time(filename, title):
    # Read the lines of the file
    with open(filename, "r") as f:
        lines = f.readlines()
    
    # Extract the frequent itemsets and execution time
    frequent_itemsets = []
    execution_time = None
    for line in lines:
        if line.startswith("Frequent itemsets:"):
            continue
        elif line.startswith("Execution time:"):
            execution_time = float(line.split(":")[1].replace("seconds", "").strip())
        else:
            parts = line.split(":")
            support = float(parts[0].strip())
            itemset = frozenset(parts[1].strip().split())
            frequent_itemsets.append((itemset, support))
    
    # Extract the minimum support from the filename
    min_support = extract_min_support(filename)
    
    # Plot the execution time vs minimum support
    if min_support is not None and execution_time is not None:
        plt.plot(min_support, execution_time, marker="o")
        plt.xlabel("Minimum Support")
        plt.ylabel("Execution Time (Seconds)")
        plt.title("{} Execution Time vs Minimum Support".format(title))


# Iterate over output files and plot execution times
for filename in os.listdir("output"):
    if filename.startswith("apriori"):
        plot_execution_time(os.path.join("output", filename), "Apriori")
    elif filename.startswith("fp_growth"):
        plot_execution_time(os.path.join("output", filename), "FP-Growth")
    # elif filename.startswith("eclat"):
    #     plot_execution_time(os.path.join("output", filename), "ECLAT")

# Show the plot
plt.show()