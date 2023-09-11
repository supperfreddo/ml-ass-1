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

# distibution of a feature
plt.hist(df['rarity'], 8)
plt.xlabel('rarity')
plt.ylabel('frequency')
plt.show()

input("\nPress Enter to continue...")

# remove rows with missing values in column power, toughness
df = df.dropna(subset=['power', 'toughness'])

# convert rows with string values to flot in column power, toughness
df['power'] = pd.to_numeric(df['power'], errors='coerce')

# print amount of rows after deleting missing values
print("Number of samples after deleting missing values: "+str(df.shape[0]))

# correlations
print("\nCORRELATIONS:")
print(df[['power', 'toughness']].corr())

input("\nPress Enter to continue...")

#### To Do

#### Scaling: z-score normalization
# print("\nSCALING")
# from sklearn import preprocessing
# scaler = preprocessing.StandardScaler()
# df_scaled = scaler.fit_transform(df[['power', 'toughness']])
# df_scaled = pd.DataFrame(df_scaled, columns=df.columns[0:4])

# print("\nMeans of original data:")
# print(df.mean())
# print("\nStandard deviations of original data:")
# print(df.std())

# print("\nMeans of transformed data:")
# print(df_scaled.mean())
# print("\nStandard deviations of transformed data:")
# print(df_scaled.std())