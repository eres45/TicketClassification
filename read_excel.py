import pandas as pd

data = pd.read_excel('ai_dev_assignment_tickets_complex_1000.xls')

print("First 5 rows of the dataset:")
print(data.head())

print("\nColumn names:")
print(data.columns.tolist())

print("\nBasic information:")
print(data.info())

print("\nBasic statistics:")
print(data.describe(include='all'))
