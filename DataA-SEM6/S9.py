import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Create the transactions dataset
transactions = [
    ['milk', 'bread', 'eggs'],
    ['milk', 'bread', 'butter', 'jam'],
    ['eggs', 'butter', 'jam'],
    ['milk', 'bread', 'eggs', 'butter'],
    ['bread', 'butter']
]

# Convert the transactions dataset into a DataFrame
df = pd.DataFrame(transactions)

# Display information about the dataset
print("Dataset Information:")
print(df.info())

# Convert categorical values into numeric format using one-hot encoding
df_encoded = pd.get_dummies(df.apply(lambda x: pd.Series(x), axis=1).stack()).sum(level=0)

# Apply the Apriori algorithm
min_support = 0.4
frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display frequent itemsets and association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
