import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Create the transactions dataset
transactions = [
    ['milk', 'bread', 'eggs'],
    ['bread', 'butter', 'cheese'],
    ['eggs', 'butter', 'yogurt'],
    ['milk', 'bread', 'butter', 'cheese'],
    ['eggs', 'bread', 'yogurt']
]

# Convert the dataset into a DataFrame
df = pd.DataFrame(transactions)

# Convert categorical values into numeric format using one-hot encoding
df_encoded = pd.get_dummies(df.apply(pd.Series).stack()).sum(level=0)

# Apply the Apriori algorithm
min_support = 0.2
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display frequent itemsets and association rules
print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
