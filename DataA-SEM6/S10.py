import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Create the dataset
data = {
    'TID': [1, 2, 3, 4, 5],
    'Items': [['eggs', 'milk', 'bread'],
              ['eggs', 'apple'],
              ['milk', 'bread'],
              ['apple', 'milk'],
              ['milk', 'apple', 'bread']]
}

# Convert the dataset into a DataFrame
df = pd.DataFrame(data)

# Convert categorical values into numeric format using one-hot encoding
df_encoded = pd.get_dummies(df['Items'].apply(pd.Series).stack()).sum(level=0)

# Apply the Apriori algorithm with different min_sup values
min_support_values = [0.2, 0.3, 0.4]

for min_support in min_support_values:
    print(f"\nMin Support: {min_support}")
    
    # Apply Apriori algorithm
    frequent_itemsets = apriori(df_encoded, min_support=min_support, use_colnames=True)
    
    # Generate association rules
    association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
    
    # Display frequent itemsets and association rules
    print("Frequent Itemsets:")
    print(frequent_itemsets)
    
    print("\nAssociation Rules:")
    print(association_rules_df)
