import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from mlxtend.frequent_patterns import apriori
import pyECLAT

h_dataset_files = ["T25I10D010K-h.txt", "T10I04D100K-h.txt"]
v_dataset_files = ["T25I10D010K-v.txt", "T10I04D100K-v.txt"]

# List the minimum support thresholds
min_support_thresholds = [0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.001]
h_algorithms = ["fpgrowth", "apriori"]
v_algorithms = ["eclat"]

# for algorithm in h_algorithms:
#     # Iterate over min_supports
#     for min_support in min_support_thresholds:
#         #iterate over h_datasets
#         for i, h_file in enumerate(h_dataset_files):
#             output_file = "output/{}_{}_{}".format(algorithm, i, min_support)
#             with open(h_file, "r") as f:
#                 transactions = [line.strip().split() for line in f]

#             te = TransactionEncoder()
#             te_ary = te.fit(transactions).transform(transactions)
#             df = pd.DataFrame(te_ary, columns=te.columns_)

#             # Run algorithms and time them
#             start_time = time.time()
#             if algorithm == "apriori":
#                 frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
#             elif algorithm == "fpgrowth":
#                 frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
#             execution_time = time.time() - start_time

#             # Write the frequent itemsets and execution time to the output file
#             with open(output_file, "w") as f:
#                 f.write("Frequent itemsets:\n")
#                 for index, row in frequent_itemsets.iterrows():
#                     f.write(str(row['support']) + " : " + str(row['itemsets']) + "\n")
#                 f.write("Execution time: " + str(execution_time) + " seconds")

for algorithm in v_algorithms:
    # Iterate over min_supports
    for min_support in min_support_thresholds:
        for i, dataset_file in enumerate(v_dataset_files):
            # Read the transaction data from the file
            with open(dataset_file, "r") as f:
                transactions = [line.strip().split() for line in f]

            # Convert the transaction data to a binary dataframe
            df = pd.DataFrame(0, index=range(len(transactions)), columns=set.union(*[set(t) for t in transactions]))
            for j, transaction in enumerate(transactions):
                for item in transaction:
                    df.at[j, item] = 1

            # Initialize the Eclat object and time the computation
            eclat = pyECLAT.ECLAT(data=df, verbose=False)
            start_time = time.time()
            eclat.fit(min_support=min_support, min_combination=1, max_combination=len(df.columns))
            frequent_itemsets = eclat.items_
            execution_time = time.time() - start_time

            # Write the frequent itemsets and execution time to the output file
            output_file = "output/{}_{}_{}.txt".format(algorithm, i, min_support)
            with open(output_file, "w") as f:
                f.write("Frequent itemsets:\n")
                for itemset in frequent_itemsets:
                    support = itemset[1] / len(transactions)
                    itemset_str = " ".join(sorted(itemset[0].split(" & ")))
                    f.write("{:.12f} : {}\n".format(support, itemset_str))
                f.write("Execution time: {:.6f} seconds".format(execution_time))


