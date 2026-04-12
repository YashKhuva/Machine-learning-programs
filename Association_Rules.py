import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from datetime import datetime, UTC

datetime.now(UTC)

dataset = [['milk', 'bread', 'butter'],
           ['bread', 'butter'],
           ['milk', 'bread', 'butter', 'cheese'],
           ['milk', 'bread'],
           ['butter', 'cheese'],
           ['bread', 'butter', 'cheese']]

te = TransactionEncoder()
te_data = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_data, columns=te.columns_)

frequent_itemsets = apriori(df, min_support=0.5, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.75)

print("Frequent Itemsets:")
print(frequent_itemsets)
print("\nAssociation Rules:")
print(rules)
