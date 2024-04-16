import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

# Creating the dataset
transaction =[['eggs', 'milk', 'bread'],
               ['eggs', 'apple'],
               ['milk', 'bread'],
               ['apple', 'milk'],
               ['milk', 'apple', 'bread']]

# Converting categorical values into numeric format
te = TransactionEncoder()
ta = te.fit(transaction).transform(transaction)
df = pd.DataFrame(ta, columns=te.columns_)

# Applying Apriori algorithm to generate frequent itemsets
min_support_values = [0.2, 0.3, 0.4]  # Different min_sup values
for min_support in min_support_values:
    print(f"\nMin Support: {min_support}")
    freq = apriori(df, min_support=min_support, use_colnames=True)
    print("Frequent Itemsets:")
    print(freq)

    # Generating association rules
    rules = association_rules(freq, metric="confidence", min_threshold=0.5)
    print("\nAssociation Rules:")
    print(rules)
