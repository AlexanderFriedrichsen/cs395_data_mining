from mlxtend.frequent_patterns import association_rules
import pandas as pd
import os

output_folder = "output"

for filename in os.listdir(output_folder):
    # Load the frequent itemsets from the output file
    with open(output_folder + "/" + filename, 'r') as f:
        itemsets = []
        for line in f:
            line = line.strip()
            if line.startswith('0.'):  # Check if the line starts with a support value
                support, items = line.split(' : ')
                items = frozenset(eval(items))
                itemsets.append(items)


    # Loop over multiple min_conf values
    min_conf_values = [0.2, 0.3, 0.4]
    min_sup_value = filename.split("_")[2]
    for min_conf in min_conf_values:
        print(f'Association rules with min_conf={min_conf}:')
        
        # Generate association rules with confidence above min_conf
        rules = association_rules(itemsets, metric='confidence', min_threshold=min_conf)
        rules = rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']]
        rules = rules.sort_values(by='lift', ascending=False)
        rules = rules.head(20)
        
        print(rules)