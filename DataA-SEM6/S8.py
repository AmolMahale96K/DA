import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Step 1: Read the dataset
dataset_path = 'market_basket_dataset.csv'
data = pd.read_csv(dataset_path)

# Step 2: Display dataset information
print("Dataset Information:")
print(data.info())

# Step 3: Preprocess the data
# Drop null values
data.dropna(inplace=True)

# Step 4: Convert categorical values into numeric format
# Assuming the dataset has categorical columns, you can use one-hot encoding
data_encoded = pd.get_dummies(data)

# Step 5: Apply the Apriori algorithm
min_support = 0.05  # Adjust as needed
frequent_itemsets = apriori(data_encoded, min_support=min_support, use_colnames=True)

# Generate association rules
association_rules_df = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Display frequent itemsets and association rules
print("\nFrequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(association_rules_df)
