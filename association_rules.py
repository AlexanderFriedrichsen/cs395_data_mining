import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules
import pyECLAT

h_dataset_files = ["T25I10D010K-h.txt", "T10I04D100K-h.txt"]
v_dataset_files = ["T25I10D010K-v.txt", "T10I04D100K-v.txt"]

# List the minimum support thresholds
min_support_thresholds =  [0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.001] 
h_algorithms = ["eclat"]#["fpgrowth", "apriori", "eclat"]
#v_algorithms = ["eclat"]

for algorithm in h_algorithms:
    # Iterate over min_supports
    for min_support in min_support_thresholds:
        #iterate over h_datasets
        for i, h_file in enumerate(h_dataset_files):
            output_file = "output/{}_{}_{}".format(algorithm, i, min_support)
            with open(h_file, "r") as f:
                if algorithm == "eclat":
                    transactions = [line.strip().split("\t") for line in f]
                    df = pd.DataFrame(transactions)
                    #print(df.head())
                else:
                    transactions = [line.strip().split() for line in f]
                    te = TransactionEncoder()
                    te_ary = te.fit(transactions).transform(transactions)
                    df = pd.DataFrame(te_ary, columns=te.columns_)

            # Run algorithms and time them
            start_time = time.time()
            if algorithm == "apriori":
                frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
            elif algorithm == "fpgrowth":
                frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
            elif algorithm == "eclat":
                eclat_instance = pyECLAT.ECLAT(df, verbose=False)
                eclat_instance.df_bin
                frequent_itemsets = eclat_instance.fit(min_support=min_support)
                print(type(frequent_itemsets))
                print(frequent_itemsets)
            execution_time = time.time() - start_time
            
            min_conf_values = [0.2, 0.3, 0.4]
            for min_conf in min_conf_values:
                a_rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_conf)
                if len(a_rules) <= 20 and len(a_rules) > 4:
                    output_file_a = "output/arules_{}_{}_{}_{}".format(algorithm, i, min_conf, min_support)
                    a_rules.to_csv(output_file_a)
                    # with open(output_file_a, "w") as f:
                        
                    #     for rule in a_rules:
                    #        print(a_rules.head)
                    #         f.write(str(rule)+ "\n")

            # Write the frequent itemsets and execution time to the output file
            with open(output_file, "w") as f:
                for index, row in frequent_itemsets.iterrows():
                    f.write(str(row['support']) + " : " + str(row['itemsets']) + "\n")
                f.write("Execution time: " + str(execution_time) + " seconds")

# for algorithm in v_algorithms:
#     # Iterate over min_supports
#     for min_support in min_support_thresholds:
#         for i, dataset_file in enumerate(h_dataset_files):



#             # Initialize the Eclat object and time the computation
#             eclat_instance = pyECLAT.ECLAT(df, verbose=False)
#             start_time = time.time()
            
#             frequent_itemsets = eclat_instance.fit(min_support=min_support)

#             execution_time = time.time() - start_time
#             # Print the frequent itemsets
#             for itemset in frequent_itemsets:
#                 print(itemset)

            # # Write the frequent itemsets and execution time to the output file
            # output_file = "output/{}_{}_{}.txt".format(algorithm, i, min_support)
            # with open(output_file, "w") as f:
            #     f.write("Frequent itemsets:\n")
            #     for itemset in frequent_itemsets:
            #         support = itemset[1] / len(transactions)
            #         itemset_str = " ".join(sorted(itemset[0].split(" & ")))
            #         f.write("{:.12f} : {}\n".format(support, itemset_str))
            #     f.write("Execution time: {:.6f} seconds".format(execution_time))

