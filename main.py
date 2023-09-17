import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#### 2. Load data
df = pd.read_csv('data/all_mtg_cards.csv', header = 0)

#### Some data exploration
print("DATA EXPLORATION")
input("\nPress Enter to continue...")

# 3. Display the first few rows of the dataset
print("\nFirst few rows:")
print(df.head())

# 4. Get basic information about the dataset
print("\nBasic info:")
print(df.info())

# 5. Get summary statistics for numeric columns
print("\nSummary statistics:")
print(df.describe())

# 6. Get dimensions / number of rows and columns in the dataset
print("Number of rows:", df.shape[0])
print("Number of columns:", df.shape[1])

# 7.Check for missing values
print("Missing values:")
print(df.isna().sum())

# 8. Distribution of a feature
plt.hist(df['rarity'], 8)
plt.xlabel('rarity')
plt.ylabel('frequency')
plt.show()
print("\nDistribution of rarity:")
print(df['rarity'].value_counts(normalize=True)) # normalize=True to get relative frequencies

# # # # TO BE REMOVED??
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

input("\nPress Enter to continue...")
#### Data Preprocessing
print("\nDATA PREPROCESSING:")
# 9. Convert rows with string values to float in column power, toughness, cmc
df['power'] = pd.to_numeric(df['power'], errors='coerce')
df['toughness'] = pd.to_numeric(df['toughness'], errors='coerce')
df['cmc'] = pd.to_numeric(df['cmc'], errors='coerce')

# Remove rows with missing values in column power, toughness, cmc
df = df.dropna(subset=['power', 'toughness', 'cmc'])

# Print amount of rows after deleting missing values
print("Number of samples after deleting missing values: "+str(df.shape[0]))

# 10. Correlations
print("\nCORRELATIONS:")
print(df[['power', 'toughness', 'cmc']].corr())

input("\nPress Enter to continue...")

# 11. Scaling: z-score normalization
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

#### knn
print("\nKNN:")
# build a kNN model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(df_scaled.values, df['rarity'])

# use the model to predict new example
predicted = knn.predict([[2, 1, 30]])
print(predicted)

#### Naive Bayes
print("\nNAIVE BAYAS:")
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
