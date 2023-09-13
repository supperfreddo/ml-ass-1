import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

#### Some data exploration
# data size
print("DATA EXPLORATION")
print("Data sample:")
print(df[0:5])

print("\nFeatures:")
for c in df.columns:
    print("\t"+c)

print("\nNumber of samples: "+str(df.shape[0]))

input("\nPress Enter to continue...")

# Display the first few rows of the dataset
print("\nFirst few rows:")
print(df.head())

# Get basic information about the dataset
print("\nBasic info:")
print(df.info())

# Get summary statistics for numeric columns
print("\nSummary statistics:")
print(df.describe())

# Get the number of rows and columns in the dataset
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# # Check for missing values
# print("Missing values:")
# print(df.isna().sum())

# # Create a mask to filter cards with rarity 'Rare'
# rare_mask = df['rarity'] == 'Rare'
# rare_cards = df[rare_mask]
# print("Rare Cards:")
# print(rare_cards.head())

# # Find unique card types
# unique_card_types = np.unique(df['type'])
# print("Unique Card Types:")
# print(unique_card_types)

# # Calculate the correlation matrix between numeric attributes
# correlation_matrix = np.corrcoef(df[['cmc', 'power', 'toughness']], rowvar=False)
# print("Correlation Matrix:")
# print(correlation_matrix)

# distibution of a feature
plt.hist(df['rarity'], 8)
plt.xlabel('rarity')
plt.ylabel('frequency')
plt.show()

input("\nPress Enter to continue...")

# convert rows with string values to flot in column power, toughness, cmc
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')
df['cmc'] = pd.to_numeric(df['cmc'], errors='coerce')

# remove rows with missing values in column power, toughness, cmc
df = df.dropna(subset=['power', 'toughness', 'cmc'])

# print amount of rows after deleting missing values
print("Number of samples after deleting missing values: "+str(df.shape[0]))

# correlations
print("\nCORRELATIONS:")
print(df[['power', 'toughness', 'cmc']].corr())

input("\nPress Enter to continue...")

#### Scaling: z-score normalization
print("\nSCALING")
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
df_scaled = scaler.fit_transform(df[['power', 'toughness', 'cmc']])
# 5 = cmc, 16 = power, 17 = toughness
df_scaled = pd.DataFrame(df_scaled, columns=df.columns[[5, 16, 17]])


#### To Do
# print("\nMeans of original data:")
# print(df.mean())
# print("\nStandard deviations of original data:")
# print(df.std())

print("\nMeans of transformed data:")
print(df_scaled.mean())
print("\nStandard deviations of transformed data:")
print(df_scaled.std())



#### Naive Bayes
print("\nNaive Bayes:")
# import libraries
from sklearn.naive_bayes import GaussianNB

# split data into features and target
X = df[['power', 'toughness', 'cmc']]
y = df['rarity']

# build a Naïve Bayes model
clf = GaussianNB()
clf.fit(X.values, y)


# use the model to predict new example
predicted = clf.predict([[2, 1, 30]])
print(predicted)