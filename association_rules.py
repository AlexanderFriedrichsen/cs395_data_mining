import time
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

h_dataset_files = ["T25I10D010K-h.txt", "T10I04D100K-h.txt"]
v_dataset_files = ["T25I10D010K-v.txt", "T10I04D100K-v.txt"]

# List the minimum support thresholds
min_support_thresholds = [0.030, 0.025, 0.020, 0.015, 0.010, 0.005, 0.001]

# Iterate over min_supports
for min_support in min_support_thresholds:
    #iterate over h_datasets
    for i, h_file in enumerate(h_dataset_files):
        output_file = "apriori_{}_{}".format(i, min_support)
        with open(h_file, "r") as f:
            transactions = [line.strip().split() for line in f]

        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        df = pd.DataFrame(te_ary, columns=te.columns_)

        # Run the Apriori algorithm and time it
        start_time = time.time()
        frequent_itemsets = apriori(df, min_support=min_support, use_colnames=True)
        execution_time = time.time() - start_time

        # Write the frequent itemsets and execution time to the output file
        with open(output_file, "w") as f:
            f.write("Frequent itemsets:\n")
            for index, row in frequent_itemsets.iterrows():
                f.write(str(row['support']) + " : " + str(row['itemsets']) + "\n")
            f.write("Execution time: " + str(execution_time) + " seconds")

    # for v_file in v_dataset_files:
    #     transactions = {}
    #     for line in f:
    #         item, transaction_id = line.strip().split()
    #         if transaction_id not in transactions:
    #             transactions[transaction_id] = set()
    #         transactions[transaction_id].add(item)
    #     transactions = [list(items) for items in transactions.values()]


