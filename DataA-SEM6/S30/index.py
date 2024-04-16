import pandas as pd
import numpy as np
from mlxtend.frequent_patterns import apriori , association_rules
from mlxtend.preprocessing import TransactionEncoder

transaction =[['eggs', 'milk','bread'],['eggs','apple'], ['milk', 'bread'], ['apple','milk'], ['milk', 'apple', 'bread']]
transaction

te = TransactionEncoder()
ta = te.fit(transaction).transform(transaction)
df = pd.DataFrame(ta,columns=te.columns_)
df


freq = apriori(df,min_support=0.4,use_colnames=True)
freq